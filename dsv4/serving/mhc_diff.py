"""Differentiable fused mHC residual ops: closed-form Pallas backwards.

The forwards (``mhc.py``) are pure-bandwidth ops over the ``hc x d``
residual stream; their backwards are too, so each gets a single fused
Pallas kernel that reads every operand exactly once:

  pre_norm bwd  (recomputes mix = Σ pre_i·X_i from X — no saved residual):
      r     = rsqrt(mean(mix²) + eps)
      dmix  = r·dh − (r³/d)·⟨dh, mix⟩·mix          # RMSNorm backprop
      dX_i  = pre_i · dmix
      dpre_i = ⟨X_i, dmix⟩

  update bwd:
      dX_i      = Σ_j comb[j,i] · dXn_j
      dcomb[j,i] = ⟨dXn_j, X_i⟩
      dpost_j   = ⟨dXn_j, f⟩
      df        = Σ_j post_j · dXn_j

Everything is fp32 inside the kernels; the hc loops are static unrolls.
Per-token coefficient grads come out with the same trailing-singleton
layout the forwards use (no illegal reshapes), squeezed in the wrappers.

custom_vjp lives on inner array-only closures cached by static config
only (the repo's custom_vjp tracer rule).
"""

from __future__ import annotations

from functools import lru_cache, partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from .config import ServingTiles
from .mhc import mhc_pre_norm, mhc_update


# ---------------------------------------------------------------------------
# pre_norm backward
# ---------------------------------------------------------------------------


def _pre_norm_bwd_kernel(x_ref, pre_ref, dh_ref, dx_ref, dpre_ref,
                         *, hc: int, eps: float):
    # x: [1,BN,hc,d]  pre: [1,BN,hc,1]  dh: [1,BN,d]
    xs = [x_ref[0, :, i, :].astype(jnp.float32) for i in range(hc)]
    mix = xs[0] * pre_ref[0, :, 0, :]
    for i in range(1, hc):
        mix += xs[i] * pre_ref[0, :, i, :]
    d = mix.shape[-1]
    r = jax.lax.rsqrt(jnp.mean(mix * mix, -1, keepdims=True) + eps)  # [BN,1]
    dh = dh_ref[0].astype(jnp.float32)                               # [BN,d]
    dot = jnp.sum(dh * mix, -1, keepdims=True)                       # [BN,1]
    dmix = r * dh - (r ** 3 / d) * dot * mix
    for i in range(hc):
        dx_ref[0, :, i, :] = (pre_ref[0, :, i, :] * dmix).astype(dx_ref.dtype)
        dpre_ref[0, :, i, :] = jnp.sum(xs[i] * dmix, -1, keepdims=True)


def _pre_norm_bwd(x, pre4, dh, *, eps, bn, interpret):
    B, n, hc, d = x.shape
    dx, dpre4 = pl.pallas_call(
        partial(_pre_norm_bwd_kernel, hc=hc, eps=eps),
        grid=(B, n // bn),
        in_specs=[
            pl.BlockSpec((1, bn, hc, d), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, bn, hc, 1), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, bn, d), lambda b, t: (b, t, 0)),
        ],
        out_specs=[
            pl.BlockSpec((1, bn, hc, d), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, bn, hc, 1), lambda b, t: (b, t, 0, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((B, n, hc, d), x.dtype),
            jax.ShapeDtypeStruct((B, n, hc, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")),
        interpret=interpret,
    )(x, pre4, dh)
    return dx, dpre4


# ---------------------------------------------------------------------------
# update backward
# ---------------------------------------------------------------------------


def _update_bwd_kernel(x_ref, comb_ref, post_ref, f_ref, dxn_ref,
                       dx_ref, dcomb_ref, dpost_ref, df_ref, *, hc: int):
    xs = [x_ref[0, :, i, :].astype(jnp.float32) for i in range(hc)]
    dxn = [dxn_ref[0, :, j, :].astype(jnp.float32) for j in range(hc)]
    f = f_ref[0].astype(jnp.float32)

    df = dxn[0] * post_ref[0, :, 0, :]
    for j in range(1, hc):
        df += dxn[j] * post_ref[0, :, j, :]
    df_ref[0] = df.astype(df_ref.dtype)

    for j in range(hc):
        dpost_ref[0, :, j, :] = jnp.sum(dxn[j] * f, -1, keepdims=True)
        for i in range(hc):
            dcomb_ref[0, :, j * hc + i, :] = jnp.sum(
                dxn[j] * xs[i], -1, keepdims=True)

    for i in range(hc):
        acc = dxn[0] * comb_ref[0, :, 0 * hc + i, :]
        for j in range(1, hc):
            acc += dxn[j] * comb_ref[0, :, j * hc + i, :]
        dx_ref[0, :, i, :] = acc.astype(dx_ref.dtype)


def _update_bwd(x, comb4, post4, f, dxn, *, bn, interpret):
    B, n, hc, d = x.shape
    return pl.pallas_call(
        partial(_update_bwd_kernel, hc=hc),
        grid=(B, n // bn),
        in_specs=[
            pl.BlockSpec((1, bn, hc, d), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, bn, hc * hc, 1), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, bn, hc, 1), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, bn, d), lambda b, t: (b, t, 0)),
            pl.BlockSpec((1, bn, hc, d), lambda b, t: (b, t, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((1, bn, hc, d), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, bn, hc * hc, 1), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, bn, hc, 1), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, bn, d), lambda b, t: (b, t, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((B, n, hc, d), x.dtype),
            jax.ShapeDtypeStruct((B, n, hc * hc, 1), jnp.float32),
            jax.ShapeDtypeStruct((B, n, hc, 1), jnp.float32),
            jax.ShapeDtypeStruct((B, n, d), f.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")),
        interpret=interpret,
    )(x, comb4, post4, f, dxn)


# ---------------------------------------------------------------------------
# custom_vjp surfaces
# ---------------------------------------------------------------------------


def _pad_n(arrs, bn):
    n = arrs[0].shape[1]
    pad = (-n) % bn
    if pad == 0:
        return arrs, n
    out = []
    for a in arrs:
        cfg = [(0, 0)] * a.ndim
        cfg[1] = (0, pad)
        out.append(jnp.pad(a, cfg))
    return out, n


@lru_cache(maxsize=None)
def _make_pre_norm(eps: float, bn: int, interpret: bool):
    tiles = ServingTiles(mhc_bn=bn, interpret=interpret)

    @jax.custom_vjp
    def f(x, pre):
        return mhc_pre_norm(x, pre, eps=eps, tiles=tiles)

    def fwd(x, pre):
        return mhc_pre_norm(x, pre, eps=eps, tiles=tiles), (x, pre)

    def bwd(res, dh):
        x, pre = res
        (xp, prep, dhp), n = _pad_n(
            (x, pre.astype(jnp.float32)[..., None], dh), bn)
        dx, dpre4 = _pre_norm_bwd(xp, prep, dhp, eps=eps, bn=bn,
                                  interpret=interpret)
        return dx[:, :n], dpre4[:, :n, :, 0].astype(pre.dtype)

    f.defvjp(fwd, bwd)
    return f


@lru_cache(maxsize=None)
def _make_update(bn: int, interpret: bool):
    tiles = ServingTiles(mhc_bn=bn, interpret=interpret)

    @jax.custom_vjp
    def f(x, comb, post, f_out):
        return mhc_update(x, comb, post, f_out, tiles=tiles)

    def fwd(x, comb, post, f_out):
        return mhc_update(x, comb, post, f_out, tiles=tiles), (x, comb, post, f_out)

    def bwd(res, dxn):
        x, comb, post, f_out = res
        B, n, hc, d = x.shape
        comb4 = comb.astype(jnp.float32).reshape(B, n, hc * hc)[..., None]
        post4 = post.astype(jnp.float32)[..., None]
        (xp, comb4p, post4p, fp, dxnp), n0 = _pad_n(
            (x, comb4, post4, f_out, dxn), bn)
        dx, dcomb4, dpost4, df = _update_bwd(xp, comb4p, post4p, fp, dxnp,
                                             bn=bn, interpret=interpret)
        dcomb = dcomb4[:, :n0, :, 0].reshape(B, n0, hc, hc).astype(comb.dtype)
        return (dx[:, :n0], dcomb, dpost4[:, :n0, :, 0].astype(post.dtype),
                df[:, :n0])

    f.defvjp(fwd, bwd)
    return f


def mhc_pre_norm_diff(x, pre, *, eps: float = 1e-6,
                      tiles: ServingTiles | None = None):
    """Differentiable fused ``RMSNorm(pre · X)``."""
    if tiles is None:
        from .config import tiles_for
        tiles = tiles_for()
    return _make_pre_norm(eps, tiles.mhc_bn, tiles.interpret)(x, pre)


def mhc_update_diff(x, comb, post, f_out, *,
                    tiles: ServingTiles | None = None):
    """Differentiable fused ``comb·X + post ⊗ f``."""
    if tiles is None:
        from .config import tiles_for
        tiles = tiles_for()
    return _make_update(tiles.mhc_bn, tiles.interpret)(x, comb, post, f_out)
