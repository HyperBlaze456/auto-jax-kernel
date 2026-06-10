"""Differentiable grouped FP8 expert FFN: forward kernels + efficient backward.

The diffable unit is the **whole expert FFN** (Linear-1 + SwiGLU + fp8 cast
+ Linear-2), not the individual GEMMs. That choice is what makes the
backward memory-lean: the only saved residuals are the two fp8 payloads
the forward produced anyway —

    x_q  : ActQuant [M, K]    (~1.13 B/elem incl. scales)
    h_q  : ActQuant [M, dff]  (the fused SwiGLU output)

versus the ~12 B/elem a naive bf16 chain would checkpoint (x, gate, up,
h). Gate/up are *recomputed* in the backward from x_q·W13 — one extra
grouped-GEMM pass, the FlashAttention recompute trade applied to the FFN.

Backward dataflow (per expert group, all fp32 accumulation):

    dh   = gmm_dgrad(dy, W2)                  # [M, dff]   Pallas
    dW2  = tgmm(dequant(h_q)ᵀ, dy)            # [E, dff, N] megablox (bf16)
    d13  = swiglu_bwd(x_q, W13, dh)           # [M, 2dff]  Pallas (recompute g,u)
    dx   = gmm_dgrad(d13, W13)                # [M, K]     Pallas
    dW13 = tgmm(xᵀ, d13)                      # [E, K, 2dff] megablox

Precision policy (DSv3 recipe): forward GEMMs run the fp8 block-scaled
path; **gradients flow in bf16 with fp32 accumulation** (no fp8 grads).
Quantization (acts and weights) backpropagates as straight-through (STE),
matching the paper's QAT formulation ("gradients are computed with respect
to the same FP8 weights used in the forward") — the test oracle encodes
the identical STE so agreement is exact up to reassociation.

The dgrad kernel mirrors the forward's layout discipline: contraction
steps over N in 128-blocks, weight scales entering as lane-broadcast rows
from a *transposed* broadcast table ``s_bcast_t[E, N/128, 1, K]`` — every
scale lookup is a BlockSpec index_map, no dynamic lane indexing.

Determinism: dgrad/swiglu-bwd are one-producer-per-element with fixed
contraction order; tgmm accumulates per-group tiles in a fixed grid order
— no atomics anywhere (paper §3.3, structurally).
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import NamedTuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.megablox.gmm import tgmm

from .config import QBLOCK
from .gemm_fp8 import (
    GemmWeight,
    _common_specs,
    _dot_block,
    _store_mask,
    gmm_fp8,
    gmm_fp8_swiglu_quant,
    prepare_weight,
)
from .quant import (
    ActQuant,
    WeightQuant,
    dequantize_act,
    dequantize_weight,
    quantize_act,
    quantize_weight,
)


class GemmWeightBwd(NamedTuple):
    """dgrad-ready weight view: same fp8 payload, scales broadcast along K.

    ``s_bcast_t[..., nb, 0, k] = s[..., k // 128, nb]`` — the transpose of
    the forward's broadcast table, so the dgrad kernel (which contracts
    over N and outputs K lanes) applies scales as a single ``[1, K]`` row
    multiply per N-step.
    """

    q: jax.Array          # [..., K, N] e4m3 (shared with the fwd view)
    s_bcast_t: jax.Array  # [..., N/QBLOCK, 1, K] f32


def prepare_weight_bwd(wq: WeightQuant) -> GemmWeightBwd:
    s_t = jnp.swapaxes(wq.s, -1, -2)                       # [..., N/128, K/128]
    s_bcast_t = jnp.repeat(s_t, QBLOCK, axis=-1)[..., :, None, :]
    return GemmWeightBwd(q=wq.q, s_bcast_t=s_bcast_t)


# ---------------------------------------------------------------------------
# dgrad kernel: dX[M, K] = dY[M, N] · W_deqᵀ[N, K]   (grouped)
# ---------------------------------------------------------------------------


def _gmm_dgrad_kernel(
    group_metadata,
    dy_ref,           # [tm, QBLOCK]        bf16   (this N-block of dY)
    w_ref,            # [K, QBLOCK]         e4m3   (this expert, this N-block)
    ws_ref,           # [1, 1, K]           f32    (lane-broadcast over K)
    dx_ref,           # [tm, K]             out
    acc_ref,          # [tm, K]             f32 scratch
    *,
    n_nsteps: int,
    tm: int,
    k: int,
    compute_upcast: bool,
):
    grid_id = pl.program_id(0)
    n_i = pl.program_id(1)

    @pl.when(n_i == 0)
    def _zero_acc():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    dy = dy_ref[...]                                       # bf16 [tm, 128]
    w = w_ref[...]
    if compute_upcast:
        w = w.astype(jnp.bfloat16)
    # contract over the N-block: dy [tm, nb] · w [K, nb] → [tm, K]
    part = jax.lax.dot_general(
        dy, w, dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32)
    acc_ref[...] += part * ws_ref[...].reshape(1, k)

    @pl.when(n_i == n_nsteps - 1)
    def _store():
        mask = _store_mask(grid_id, group_metadata, tm=tm, tn=k)
        dx_ref[...] = jax.lax.select(
            mask, acc_ref[...], dx_ref[...].astype(jnp.float32)
        ).astype(dx_ref.dtype)


def gmm_dgrad(
    dy: jax.Array,            # [M, N] bf16/f32
    w_bwd: GemmWeightBwd,     # q [E, K, N]
    group_sizes: jax.Array,
    *,
    tm: int = 128,
    out_dtype=jnp.bfloat16,
    compute_upcast: bool = True,
    interpret: bool = False,
) -> jax.Array:
    """Grouped dX = dY·W_deqᵀ. Same group layout contract as ``gmm_fp8``."""
    from jax.experimental.pallas.ops.tpu.megablox.gmm import make_group_metadata

    m, n = dy.shape
    e, kw, nw = w_bwd.q.shape
    if nw != n:
        raise ValueError(f"N mismatch: dy {n} vs w {nw}")
    if m % tm != 0 or n % QBLOCK != 0 or kw % QBLOCK != 0:
        raise ValueError("M must be tm-aligned; K, N must be 128-aligned")

    group_metadata, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes.astype(jnp.int32), m=m, tm=tm,
        start_group=jnp.zeros((), jnp.int32), num_nonzero_groups=e,
        visit_empty_groups=False)

    def dy_map(g, n_i, gm):
        return gm[2][g], n_i

    def w_map(g, n_i, gm):
        return gm[1][g], 0, n_i

    def ws_map(g, n_i, gm):
        return gm[1][g], n_i, 0, 0

    return pl.pallas_call(
        partial(_gmm_dgrad_kernel, n_nsteps=n // QBLOCK, tm=tm, k=kw,
                compute_upcast=compute_upcast),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_active_tiles, n // QBLOCK),
            in_specs=[
                pl.BlockSpec((tm, QBLOCK), dy_map),
                pl.BlockSpec((None, kw, QBLOCK), w_map),
                pl.BlockSpec((None, 1, 1, kw), ws_map),
            ],
            out_specs=pl.BlockSpec((tm, kw), lambda g, ni, gm: (gm[2][g], 0)),
            scratch_shapes=[pltpu.VMEM((tm, kw), jnp.float32)],
        ),
        out_shape=jax.ShapeDtypeStruct((m, kw), out_dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary")),
        interpret=interpret,
    )(group_metadata, dy.astype(jnp.bfloat16), w_bwd.q, w_bwd.s_bcast_t)


# ---------------------------------------------------------------------------
# SwiGLU activation backward (recomputes gate/up from the saved fp8 x)
# ---------------------------------------------------------------------------


def _swiglu_bwd_kernel(
    group_metadata,
    lhs_q_ref,        # [tm, QBLOCK]      e4m3   (saved x_q — exact fwd operand)
    lhs_s_ref,        # [1, tm, 1]        f32
    rhs_q_ref,        # [QBLOCK, 2*dff]   e4m3   (W13)
    rhs_s_ref,        # [1, 1, 2*dff]     f32
    dh_ref,           # [tm, dff]         bf16   (upstream cotangent on h)
    d13_ref,          # [tm, 2*dff]       out: [dgate ‖ dup] bf16
    acc_ref,          # [tm, 2*dff]       f32 scratch (recomputed gate‖up)
    *,
    n_ksteps: int,
    tm: int,
    dff: int,
    compute_upcast: bool,
):
    grid_id = pl.program_id(0)
    k_i = pl.program_id(1)

    @pl.when(k_i == 0)
    def _zero_acc():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    part = _dot_block(lhs_q_ref[...], rhs_q_ref[...],
                      compute_upcast=compute_upcast)
    acc_ref[...] += part * lhs_s_ref[...].reshape(tm, 1) \
        * rhs_s_ref[...].reshape(1, 2 * dff)

    @pl.when(k_i == n_ksteps - 1)
    def _bwd_store():
        acc = acc_ref[...]
        g, u = acc[:, :dff], acc[:, dff:]
        sig = jax.nn.sigmoid(g)
        silu = g * sig
        dsilu = sig * (1.0 + g * (1.0 - sig))              # d/dg SiLU(g)
        dh = dh_ref[...].astype(jnp.float32)
        dg = dh * u * dsilu
        du = dh * silu
        mask = _store_mask(grid_id, group_metadata, tm=tm, tn=2 * dff)
        d13 = jnp.concatenate([dg, du], axis=1)
        d13_ref[...] = jax.lax.select(
            mask, d13, d13_ref[...].astype(jnp.float32)
        ).astype(d13_ref.dtype)


def swiglu_bwd(
    x_q: ActQuant,            # saved forward operand [M, K]
    w13: GemmWeight,          # [E, K, 2*dff]
    dh: jax.Array,            # [M, dff] bf16/f32
    group_sizes: jax.Array,
    *,
    tm: int = 128,
    compute_upcast: bool = True,
    interpret: bool = False,
) -> jax.Array:
    """Returns ``d13 = [dgate ‖ dup] [M, 2*dff]`` bf16, recomputing gate/up
    from the *exact* forward operands (x_q · W13) — no saved activations."""
    from jax.experimental.pallas.ops.tpu.megablox.gmm import make_group_metadata

    m, k = x_q.q.shape
    e, _, n2 = w13.q.shape
    dff = n2 // 2
    group_metadata, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes.astype(jnp.int32), m=m, tm=tm,
        start_group=jnp.zeros((), jnp.int32), num_nonzero_groups=e,
        visit_empty_groups=False)
    lhs_q_spec, lhs_s_spec, rhs_q_spec, rhs_s_spec, n_kblocks = _common_specs(
        tm=tm, k=k, n=n2)

    return pl.pallas_call(
        partial(_swiglu_bwd_kernel, n_ksteps=n_kblocks, tm=tm, dff=dff,
                compute_upcast=compute_upcast),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_active_tiles, n_kblocks),
            in_specs=[
                lhs_q_spec, lhs_s_spec, rhs_q_spec, rhs_s_spec,
                pl.BlockSpec((tm, dff), lambda g, ki, gm: (gm[2][g], 0)),
            ],
            out_specs=pl.BlockSpec((tm, n2), lambda g, ki, gm: (gm[2][g], 0)),
            scratch_shapes=[pltpu.VMEM((tm, n2), jnp.float32)],
        ),
        out_shape=jax.ShapeDtypeStruct((m, n2), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary")),
        interpret=interpret,
    )(group_metadata, x_q.q, x_q.s_t[..., None], w13.q, w13.s_bcast,
      dh.astype(jnp.bfloat16))


# ---------------------------------------------------------------------------
# Diffable grouped expert FFN (custom_vjp on an array-only inner factory)
# ---------------------------------------------------------------------------


def _zero_rows_beyond(x: jax.Array, n_valid: jax.Array) -> jax.Array:
    """Zero rows >= n_valid (padding rows must not contribute to wgrad)."""
    rows = jax.lax.broadcasted_iota(jnp.int32, (x.shape[0], 1), 0)
    return jnp.where(rows < n_valid, x, 0)


@lru_cache(maxsize=None)
def _make_grouped_ffn(tm: int, compute_upcast: bool, interpret: bool):
    """custom_vjp closure. Cached on static config only; every tensor
    (including int group_sizes) flows through as a traced argument —
    the custom_vjp lives on an inner array-only function (see the repo's
    custom_vjp tracer lesson)."""

    def _fwd_compute(x, w13, w2, group_sizes):
        x_q = quantize_act(x.astype(jnp.float32))
        w13_q = quantize_weight(w13)
        w2_q = quantize_weight(w2)
        h_q = gmm_fp8_swiglu_quant(
            x_q, prepare_weight(w13_q), group_sizes, tm=tm,
            compute_upcast=compute_upcast, interpret=interpret)
        y = gmm_fp8(
            h_q, prepare_weight(w2_q), group_sizes, tm=tm,
            out_dtype=jnp.bfloat16, compute_upcast=compute_upcast,
            interpret=interpret)
        return y, (x_q, h_q, w13_q, w2_q)

    @jax.custom_vjp
    def ffn(x, w13, w2, group_sizes):
        return _fwd_compute(x, w13, w2, group_sizes)[0]

    def ffn_fwd(x, w13, w2, group_sizes):
        y, (x_q, h_q, w13_q, w2_q) = _fwd_compute(x, w13, w2, group_sizes)
        # NOTE: master x is NOT a residual — the wgrad operand is the
        # *dequantized fp8 x* (the tensor the forward actually multiplied;
        # using master x here injects the act-quant error into dW13).
        # Zero-size sentinels keep the primal dtypes for cotangent casts.
        dts = (jnp.zeros((0,), x.dtype), jnp.zeros((0,), w13.dtype),
               jnp.zeros((0,), w2.dtype))
        return y, (x_q, h_q, w13_q, w2_q, group_sizes, dts)

    def ffn_bwd(res, dy):
        x_q, h_q, w13_q, w2_q, group_sizes, dts = res
        x_dt, w13_dt, w2_dt = dts
        n_valid = group_sizes.sum()
        dy = _zero_rows_beyond(dy.astype(jnp.bfloat16), n_valid)

        # dh w.r.t. the (STE-identity) hidden, via W2.
        dh = gmm_dgrad(dy, prepare_weight_bwd(w2_q), group_sizes, tm=tm,
                       compute_upcast=compute_upcast, interpret=interpret)
        # dW2 = h_deqᵀ · dY (bf16 megablox tgmm, fp32 accum).
        h_deq = _zero_rows_beyond(
            dequantize_act(h_q).astype(jnp.bfloat16), n_valid)
        dw2 = tgmm(h_deq.T, dy, group_sizes.astype(jnp.int32),
                   tiling=(tm, min(128, h_deq.shape[1]), 128),
                   interpret=interpret).astype(jnp.float32)

        # SwiGLU activation backward (recomputes gate/up from x_q·W13).
        d13 = swiglu_bwd(x_q, prepare_weight(w13_q), dh, group_sizes, tm=tm,
                         compute_upcast=compute_upcast, interpret=interpret)
        d13 = _zero_rows_beyond(d13, n_valid)

        dx = gmm_dgrad(d13, prepare_weight_bwd(w13_q), group_sizes, tm=tm,
                       compute_upcast=compute_upcast, interpret=interpret)
        x_deq = _zero_rows_beyond(
            dequantize_act(x_q).astype(jnp.bfloat16), n_valid)
        dw13 = tgmm(x_deq.T, d13, group_sizes.astype(jnp.int32),
                    tiling=(tm, min(128, x_deq.shape[1]), 128),
                    interpret=interpret).astype(jnp.float32)

        dgs = np.zeros(group_sizes.shape, jax.dtypes.float0)
        return (dx.astype(x_dt.dtype), dw13.astype(w13_dt.dtype),
                dw2.astype(w2_dt.dtype), dgs)

    ffn.defvjp(ffn_fwd, ffn_bwd)
    return ffn


def grouped_ffn(
    x: jax.Array,             # [M, K] bf16/f32, rows grouped by expert
    w13: jax.Array,           # [E, K, 2*dff] master weights (bf16/f32)
    w2: jax.Array,            # [E, dff, K_out]
    group_sizes: jax.Array,   # [E] int32
    *,
    tm: int = 128,
    compute_upcast: bool = True,
    interpret: bool = False,
) -> jax.Array:
    """Differentiable grouped expert FFN (fp8 forward, bf16 backward, STE
    through both quantizations). Output rows beyond ``sum(group_sizes)``
    are unspecified; their cotangents are ignored."""
    return _make_grouped_ffn(tm, compute_upcast, interpret)(
        x, w13, w2, group_sizes)


# ---------------------------------------------------------------------------
# STE oracle (what jax.grad is compared against in tests)
# ---------------------------------------------------------------------------


def _ste(x_deq, x):
    return x + jax.lax.stop_gradient(x_deq - x)


def grouped_ffn_ref(x, w13, w2, group_sizes):
    """Eager STE-modeled reference: identical quantization points, dense
    per-group einsums. ``jax.grad`` of this is the gradient oracle."""
    xf = x.astype(jnp.float32)
    x_ste = _ste(dequantize_act(quantize_act(xf)), xf)
    w13_ste = _ste(dequantize_weight(quantize_weight(w13)),
                   w13.astype(jnp.float32))
    w2_ste = _ste(dequantize_weight(quantize_weight(w2)),
                  w2.astype(jnp.float32))

    e = w13.shape[0]
    ends = jnp.cumsum(group_sizes)
    starts = ends - group_sizes
    rows = jnp.arange(x.shape[0])
    dff = w13.shape[-1] // 2
    out = jnp.zeros((x.shape[0], w2.shape[-1]), jnp.float32)
    for g in range(e):
        sel = ((rows >= starts[g]) & (rows < ends[g]))[:, None]
        h13 = x_ste @ w13_ste[g]
        gt, up = h13[:, :dff], h13[:, dff:]
        h = (gt * jax.nn.sigmoid(gt)) * up
        h_ste = _ste(dequantize_act(quantize_act(h)), h)
        out = out + jnp.where(sel, h_ste @ w2_ste[g], 0.0)
    return out
