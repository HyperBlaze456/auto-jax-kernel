"""Fused mHC (Manifold-Constrained Hyper-Connections) residual-stream ops.

The mHC update (paper §2.2, Eq. 1) per token:

    X_{l+1} = B_l · X_l  +  C_l · F_l(A_l · X_l)

with the residual stream ``X ∈ R^{hc x d}`` (hc=4), ``A_l = pre ∈ R^{1xhc}``,
``B_l = comb ∈ R^{hc x hc}`` (doubly stochastic via Sinkhorn), and
``C_l = post ∈ R^{hc x 1}`` — all *per-token, data-dependent* (produced by
``kernel_v2.mhc_sinkhorn_kernel_v2`` from the mixes projection).

Why fuse: the residual stream is ``hc x d`` per token — 4x wider than a
normal transformer's. At d=4096 / bf16 that is 32 KB per token per
read-or-write. The two ops here are pure bandwidth (hc=4 multiply-adds per
element — far below any compute roofline), so every avoided HBM pass is
~free time:

  - ``mhc_pre_norm`` fuses A_l·X with the layer's RMSNorm: reads X once
    (hc·d), writes h (d). Unfused, the mix intermediate would round-trip
    HBM: an extra d-write + d-read per token per layer-half.
  - ``mhc_update`` fuses B_l·X with C_l·f in one pass: reads X (hc·d) +
    f (d), writes X' (hc·d). Unfused: an extra hc·d round trip.
  - ``mhc_update_mix`` (HARDWARE_NOTES §13.6) goes one half further: the
    *next* sublayer's mixes projection (``RMSNorm_row(X') @ w_mix``, the
    input to its Sinkhorn gates) is emitted from the update's epilogue,
    while X' is still live in VMEM. Without it, the next half's gate
    path re-reads the full hc·d stream from HBM — the third and last
    X-read per half; with it, each half reads X exactly twice (pre_norm
    and update), cutting the residual-stream read traffic by ~1/3.

Across 43 layers x 2 halves (attention + MoE), the fusion saves
~(2d + hc·d) x 86 ≈ 2 MB of HBM traffic per token at the Flash config;
the epilogue emission saves a further hc·d x 86 ≈ 2.4 MB.

Layout notes: the per-token coefficient tensors (pre, post, comb) enter
with an explicit trailing singleton dim (``[..., hc, 1]`` / flattened
``[..., hc*hc, 1]``) so the kernel can slice ``[BN, 1]`` factors directly —
no 1-D→2-D reshape (illegal on the last two dims in Mosaic) is ever
needed. The hc loops are static unrolls (hc=4): everything lowers to VPU
broadcast-FMAs over ``[BN, d]`` tiles.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from .config import ServingTiles


# ---------------------------------------------------------------------------
# Kernel 1: h = RMSNorm( Σ_i pre_i · X_i )
# ---------------------------------------------------------------------------


def _pre_norm_kernel(x_ref, pre_ref, o_ref, *, hc: int, eps: float):
    # x_ref:   [1, BN, hc, d]
    # pre_ref: [1, BN, hc, 1]
    # o_ref:   [1, BN, d]
    mix = x_ref[0, :, 0, :].astype(jnp.float32) * pre_ref[0, :, 0, :]
    for i in range(1, hc):
        mix += x_ref[0, :, i, :].astype(jnp.float32) * pre_ref[0, :, i, :]
    var = jnp.mean(mix * mix, axis=-1, keepdims=True)
    o_ref[0] = (mix * jax.lax.rsqrt(var + eps)).astype(o_ref.dtype)


def mhc_pre_norm(
    x: jax.Array,        # [B, n, hc, d]
    pre: jax.Array,      # [B, n, hc]
    *,
    eps: float = 1e-6,
    tiles: ServingTiles | None = None,
) -> jax.Array:
    """Fused layer-input map: ``RMSNorm(pre · X)`` → ``[B, n, d]``."""
    if tiles is None:
        from .config import tiles_for
        tiles = tiles_for()
    B, n, hc, d = x.shape
    bn = tiles.mhc_bn
    pad = (-n) % bn
    if pad:
        x = jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)))
        pre = jnp.pad(pre, ((0, 0), (0, pad), (0, 0)))
    n_p = n + pad

    out = pl.pallas_call(
        partial(_pre_norm_kernel, hc=hc, eps=eps),
        grid=(B, n_p // bn),
        in_specs=[
            pl.BlockSpec((1, bn, hc, d), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, bn, hc, 1), lambda b, t: (b, t, 0, 0)),
        ],
        out_specs=pl.BlockSpec((1, bn, d), lambda b, t: (b, t, 0)),
        out_shape=jax.ShapeDtypeStruct((B, n_p, d), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
        interpret=tiles.interpret,
    )(x, pre.astype(jnp.float32)[..., None])
    return out[:, :n]


# ---------------------------------------------------------------------------
# Kernel 2: X' = comb · X + post ⊗ f
# ---------------------------------------------------------------------------


def _update_kernel(x_ref, comb_ref, post_ref, f_ref, o_ref, *, hc: int):
    # x_ref:    [1, BN, hc, d]
    # comb_ref: [1, BN, hc*hc, 1]   (row-major: comb[j, i] at j*hc + i)
    # post_ref: [1, BN, hc, 1]
    # f_ref:    [1, BN, d]
    # o_ref:    [1, BN, hc, d]
    f = f_ref[0].astype(jnp.float32)                       # [BN, d]
    xs = [x_ref[0, :, i, :].astype(jnp.float32) for i in range(hc)]
    for j in range(hc):
        acc = xs[0] * comb_ref[0, :, j * hc, :]
        for i in range(1, hc):
            acc += xs[i] * comb_ref[0, :, j * hc + i, :]
        acc += f * post_ref[0, :, j, :]
        o_ref[0, :, j, :] = acc.astype(o_ref.dtype)


def mhc_update(
    x: jax.Array,        # [B, n, hc, d]
    comb: jax.Array,     # [B, n, hc, hc]  (X'_j = Σ_i comb[j,i] X_i)
    post: jax.Array,     # [B, n, hc]
    f_out: jax.Array,    # [B, n, d]
    *,
    tiles: ServingTiles | None = None,
) -> jax.Array:
    """Fused residual update → ``[B, n, hc, d]``."""
    if tiles is None:
        from .config import tiles_for
        tiles = tiles_for()
    B, n, hc, d = x.shape
    bn = tiles.mhc_bn
    pad = (-n) % bn
    if pad:
        x = jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)))
        comb = jnp.pad(comb, ((0, 0), (0, pad), (0, 0), (0, 0)))
        post = jnp.pad(post, ((0, 0), (0, pad), (0, 0)))
        f_out = jnp.pad(f_out, ((0, 0), (0, pad), (0, 0)))
    n_p = n + pad

    out = pl.pallas_call(
        partial(_update_kernel, hc=hc),
        grid=(B, n_p // bn),
        in_specs=[
            pl.BlockSpec((1, bn, hc, d), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, bn, hc * hc, 1), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, bn, hc, 1), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, bn, d), lambda b, t: (b, t, 0)),
        ],
        out_specs=pl.BlockSpec((1, bn, hc, d), lambda b, t: (b, t, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((B, n_p, hc, d), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
        interpret=tiles.interpret,
    )(x,
      comb.astype(jnp.float32).reshape(B, n_p, hc * hc)[..., None],
      post.astype(jnp.float32)[..., None],
      f_out)
    return out[:, :n]


# ---------------------------------------------------------------------------
# Kernel 3: (X', mixes_next) — update + next half's mixes projection
# ---------------------------------------------------------------------------


def _update_mix_kernel(x_ref, comb_ref, post_ref, f_ref, wm_ref,
                       o_ref, mix_ref, *, hc: int, eps: float):
    # x_ref:    [1, BN, hc, d]
    # comb_ref: [1, BN, hc*hc, 1]
    # post_ref: [1, BN, hc, 1]
    # f_ref:    [1, BN, d]
    # wm_ref:   [hc, d, n_mix] f32   next half's w_mix, row-split per stream
    # o_ref:    [1, BN, hc, d]
    # mix_ref:  [1, BN, n_mix] f32
    f = f_ref[0].astype(jnp.float32)                       # [BN, d]
    xs = [x_ref[0, :, i, :].astype(jnp.float32) for i in range(hc)]
    mix = None
    for j in range(hc):
        acc = xs[0] * comb_ref[0, :, j * hc, :]
        for i in range(1, hc):
            acc += xs[i] * comb_ref[0, :, j * hc + i, :]
        acc += f * post_ref[0, :, j, :]
        xq = acc.astype(o_ref.dtype)
        o_ref[0, :, j, :] = xq
        # Epilogue: the next half's gate path would read this row back
        # from HBM and compute RMSNorm_row(X') @ w_mix — do it here while
        # the row is live. Round through the storage dtype first so the
        # operand is *elementwise identical* to that HBM read-back (the
        # only remaining difference vs the unfused path is dot-reduction
        # order, ~1 ulp f32).
        xr = xq.astype(jnp.float32)
        var = jnp.mean(xr * xr, axis=-1, keepdims=True)
        normed = xr * jax.lax.rsqrt(var + eps)             # [BN, d]
        part = jax.lax.dot_general(                        # [BN, n_mix]
            normed, wm_ref[j], (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32)
        mix = part if mix is None else mix + part
    mix_ref[0] = mix


def mhc_update_mix(
    x: jax.Array,        # [B, n, hc, d]
    comb: jax.Array,     # [B, n, hc, hc]
    post: jax.Array,     # [B, n, hc]
    f_out: jax.Array,    # [B, n, d]
    w_mix: jax.Array,    # [hc*d, n_mix] f32 — the NEXT half's projection
    *,
    eps: float = 1e-6,
    tiles: ServingTiles | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Fused residual update that also emits the next half's mixes.

    Returns ``(X' [B, n, hc, d], mixes [B, n, n_mix] f32)`` where
    ``mixes == RMSNorm_row(X').reshape(hc·d) @ w_mix`` — exactly what
    ``model._mixes_proj`` computes, minus one full hc·d HBM read.
    """
    if tiles is None:
        from .config import tiles_for
        tiles = tiles_for()
    B, n, hc, d = x.shape
    n_mix = w_mix.shape[1]
    bn = tiles.mhc_bn
    pad = (-n) % bn
    if pad:
        x = jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)))
        comb = jnp.pad(comb, ((0, 0), (0, pad), (0, 0), (0, 0)))
        post = jnp.pad(post, ((0, 0), (0, pad), (0, 0)))
        f_out = jnp.pad(f_out, ((0, 0), (0, pad), (0, 0)))
    n_p = n + pad

    out, mixes = pl.pallas_call(
        partial(_update_mix_kernel, hc=hc, eps=eps),
        grid=(B, n_p // bn),
        in_specs=[
            pl.BlockSpec((1, bn, hc, d), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, bn, hc * hc, 1), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, bn, hc, 1), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, bn, d), lambda b, t: (b, t, 0)),
            pl.BlockSpec((hc, d, n_mix), lambda b, t: (0, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((1, bn, hc, d), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, bn, n_mix), lambda b, t: (b, t, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((B, n_p, hc, d), x.dtype),
            jax.ShapeDtypeStruct((B, n_p, n_mix), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
        interpret=tiles.interpret,
    )(x,
      comb.astype(jnp.float32).reshape(B, n_p, hc * hc)[..., None],
      post.astype(jnp.float32)[..., None],
      f_out,
      w_mix.astype(jnp.float32).reshape(hc, d, n_mix))
    return out[:, :n], mixes[:, :n]


# ---------------------------------------------------------------------------
# Eager references
# ---------------------------------------------------------------------------


def mhc_pre_norm_ref(x, pre, *, eps: float = 1e-6):
    mix = jnp.einsum("bnh,bnhd->bnd", pre.astype(jnp.float32),
                     x.astype(jnp.float32))
    var = jnp.mean(mix * mix, axis=-1, keepdims=True)
    return (mix * jax.lax.rsqrt(var + eps)).astype(x.dtype)


def mhc_update_ref(x, comb, post, f_out):
    xf = x.astype(jnp.float32)
    res = jnp.einsum("bnji,bnid->bnjd", comb.astype(jnp.float32), xf)
    res += post.astype(jnp.float32)[..., None] * f_out.astype(jnp.float32)[:, :, None, :]
    return res.astype(x.dtype)


def mhc_update_mix_ref(x, comb, post, f_out, w_mix, *, eps: float = 1e-6):
    B, n, hc, d = x.shape
    xp = mhc_update_ref(x, comb, post, f_out)
    xf = xp.astype(jnp.float32)
    normed = xf * jax.lax.rsqrt((xf * xf).mean(-1, keepdims=True) + eps)
    return xp, normed.reshape(B, n, hc * d) @ w_mix.astype(jnp.float32)
