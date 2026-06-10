"""Block-scaled FP8 grouped GEMM for MoE expert compute (DeepGEMM idea, TPU form).

What this implements
--------------------

Two Pallas kernels sharing one tiling scheme:

  1. ``gmm_fp8`` — m-grouped GEMM: tokens sorted by expert in contiguous
     rows, ``out[g] = lhs[rows(g)] @ rhs[g]``, fp8 payloads, two-level
     fp32 accumulation with per-128-block scales.
  2. ``gmm_fp8_swiglu_quant`` — the same K-loop, but ``rhs`` is the fused
     W13 ``[E, K, 2*dff]`` (gate ‖ up); the epilogue applies SwiGLU and
     *re-quantizes the result to fp8 in-register*, emitting the ActQuant
     (payload + transposed scales) the W2 GEMM consumes. This is the
     paper's "SwiGLU + FP8 Cast" pipeline stage (§3.1 Figure 5) as a
     kernel epilogue: the bf16/f32 hidden tensor never exists in HBM.

HBM traffic accounting (why the tiling looks like this)
-------------------------------------------------------

The m-tile axis is the only tiled output axis; **N is not tiled** — each
program's accumulator covers the expert's full output width. Consequences:

  - expert weights stream HBM→VMEM exactly once per m-tile of that
    expert. In decode (each active expert owns ~1 tile), weights are read
    exactly once — the theoretical minimum, and the term that dominates
    decode MoE latency.
  - activations are read exactly once (no re-fetch across an n-grid).
  - the K grid axis is innermost and equals the 128-row quant block, so
    scale application needs no in-kernel dynamic lane indexing: every
    (k-step, tile) pair maps to its scales purely through BlockSpec
    index_maps.

VMEM budget at the Pro shapes (worst case, tm=128):
acc ``128x6144`` f32 (W13: 2*3072) = 3 MiB, plus double-buffered rhs tiles
``128x6144`` fp8 = 1.5 MiB, lhs tiles negligible → ~5 MiB, comfortably
inside the 32 MiB floor (v4/v6e). ``tm`` can rise to 256+ on 64 MiB parts.

Group metadata
--------------

Reuses ``make_group_metadata`` from JAX's megablox: groups need not be
tile-aligned; a tile straddling two experts is visited once per expert
(consecutively) with masked read-modify-write stores. Static grid size is
``tiles_m + E - 1``; surplus steps repeat the last tile id and are masked
out by the store mask, so no branch divergence is needed.

Numerics
--------

``out = Σ_kb (x_q[:, kb] · w_q[kb, :]) * x_s[:, kb] * w_s[kb, :]`` with the
dot in bf16 (``compute_upcast=True``; e4m3→bf16 is exact) or native fp8,
partial products in fp32, scales applied in fp32 per K-block — bit-equal
to the eager dequantized reference up to fp32 dot reassociation.

Determinism: each output element is produced by exactly one program with
a fixed sequential K order — no atomics, no cross-program accumulation
(the TPU analogue of the paper §3.3 determinism requirement comes for
free from the grid structure).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.megablox.gmm import make_group_metadata

from .config import FP8_DTYPE, FP8_MAX, QBLOCK
from .quant import ActQuant, WeightQuant

_EPS = 1e-12


class GemmWeight(NamedTuple):
    """Kernel-ready weight: fp8 payload + lane-broadcast scales.

    ``s_bcast[..., kb, 0, n] = s[..., kb, n // 128]`` — broadcasting the
    per-128x128 scale across its 128 output lanes once at weight-load time
    turns in-kernel scale application into a single ``[1, N]`` row
    multiply (no scalar extraction, no lane-dim dynamic indexing). Costs
    ``K/128 * N`` f32 per weight (~3% of the fp8 payload), paid once.
    """

    q: jax.Array          # [..., K, N] e4m3
    s_bcast: jax.Array    # [..., K/QBLOCK, 1, N] f32


def prepare_weight(wq: WeightQuant) -> GemmWeight:
    s_bcast = jnp.repeat(wq.s, QBLOCK, axis=-1)[..., :, None, :]
    return GemmWeight(q=wq.q, s_bcast=s_bcast)


def _dot_block(lhs_q, rhs_q, *, compute_upcast: bool):
    """fp8 x fp8 -> f32 MXU dot, optionally via exact bf16 upcast."""
    if compute_upcast:
        lhs_q = lhs_q.astype(jnp.bfloat16)
        rhs_q = rhs_q.astype(jnp.bfloat16)
    return jax.lax.dot_general(
        lhs_q, rhs_q,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


def _store_mask(grid_id, group_metadata, *, tm: int, tn: int):
    """Rows of the current tile that belong to the current group."""
    group_offsets, group_ids, m_tile_ids = group_metadata
    group_id = group_ids[grid_id]
    group_start = group_offsets[group_id]
    group_end = group_offsets[group_id + 1]
    m_id = m_tile_ids[grid_id] * tm
    iota = jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + m_id
    return jnp.logical_and(iota >= group_start, iota < group_end)


# ---------------------------------------------------------------------------
# Kernel bodies
# ---------------------------------------------------------------------------


def _gmm_kernel(
    group_metadata,   # scalar-prefetch: (group_offsets, group_ids, m_tile_ids)
    lhs_q_ref,        # [tm, QBLOCK]            e4m3
    lhs_s_ref,        # [1, tm, 1]              f32   (per-row scale of this k-block)
    rhs_q_ref,        # [QBLOCK, N]             e4m3  (this expert, this k-block)
    rhs_s_ref,        # [1, 1, N]               f32   (lane-broadcast)
    out_ref,          # [tm, N]                 out dtype
    acc_ref,          # [tm, N]                 f32 scratch
    *,
    n_ksteps: int,
    tm: int,
    n: int,
    compute_upcast: bool,
):
    grid_id = pl.program_id(0)
    k_i = pl.program_id(1)

    @pl.when(k_i == 0)
    def _zero_acc():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    part = _dot_block(lhs_q_ref[...], rhs_q_ref[...],
                      compute_upcast=compute_upcast)          # [tm, N] f32
    ls = lhs_s_ref[...].reshape(tm, 1)                         # row scales
    rs = rhs_s_ref[...].reshape(1, n)                          # col scales
    acc_ref[...] += part * ls * rs

    @pl.when(k_i == n_ksteps - 1)
    def _store():
        mask = _store_mask(grid_id, group_metadata, tm=tm, tn=n)
        out_ref[...] = jax.lax.select(
            mask, acc_ref[...], out_ref[...].astype(jnp.float32)
        ).astype(out_ref.dtype)


def _gmm_swiglu_quant_kernel(
    group_metadata,
    lhs_q_ref,        # [tm, QBLOCK]      e4m3
    lhs_s_ref,        # [1, tm, 1]        f32
    rhs_q_ref,        # [QBLOCK, 2*dff]   e4m3   (gate ‖ up)
    rhs_s_ref,        # [1, 1, 2*dff]     f32
    h_q_ref,          # [tm, dff]         e4m3   out
    h_s_ref,          # [dff/QBLOCK, tm, 1] f32  out (transposed ActQuant scales)
    acc_ref,          # [tm, 2*dff]       f32 scratch
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
                      compute_upcast=compute_upcast)          # [tm, 2*dff]
    ls = lhs_s_ref[...].reshape(tm, 1)
    rs = rhs_s_ref[...].reshape(1, 2 * dff)
    acc_ref[...] += part * ls * rs

    @pl.when(k_i == n_ksteps - 1)
    def _swiglu_quant_store():
        acc = acc_ref[...]
        gate = acc[:, :dff]                                     # static halves
        up = acc[:, dff:]
        h = (gate * jax.nn.sigmoid(gate)) * up                  # SiLU(gate)*up, f32

        mask = _store_mask(grid_id, group_metadata, tm=tm, tn=dff)

        # Per-1x128-group quantization, fully in-register. The j-loop is a
        # static unroll over the dff/128 lane groups.
        n_groups = dff // QBLOCK
        prev_q = h_q_ref[...].astype(jnp.float32)
        q_cols = []
        for j in range(n_groups):
            hj = h[:, j * QBLOCK:(j + 1) * QBLOCK]              # [tm, 128]
            amax = jnp.max(jnp.abs(hj), axis=1, keepdims=True)  # [tm, 1]
            sj = jnp.maximum(amax, _EPS) / FP8_MAX
            qj = jnp.clip(hj / sj, -FP8_MAX, FP8_MAX)
            q_cols.append(qj)
            # Scales land transposed ([kb, m] layout) so GEMM2 consumes
            # them with zero relayout. Masked read-modify-write keeps the
            # neighbour group's rows intact on straddled tiles.
            row_mask = mask[:, :1]                              # [tm, 1]
            h_s_ref[j] = jnp.where(row_mask, sj, h_s_ref[j])
        q_full = jnp.concatenate(q_cols, axis=1)                # [tm, dff] f32
        h_q_ref[...] = jnp.where(mask, q_full, prev_q).astype(h_q_ref.dtype)


# ---------------------------------------------------------------------------
# pallas_call wrappers
# ---------------------------------------------------------------------------


def _common_specs(*, tm: int, k: int, n: int):
    """BlockSpecs shared by both kernels. Index maps receive the grid
    coords plus the scalar-prefetch refs (group_metadata)."""
    n_kblocks = k // QBLOCK

    def lhs_q_map(grid_id, k_i, gm):
        _, _, m_tile_ids = gm
        return m_tile_ids[grid_id], k_i

    def lhs_s_map(grid_id, k_i, gm):
        _, _, m_tile_ids = gm
        return k_i, m_tile_ids[grid_id], 0

    def rhs_q_map(grid_id, k_i, gm):
        _, group_ids, _ = gm
        return group_ids[grid_id], k_i, 0

    def rhs_s_map(grid_id, k_i, gm):
        _, group_ids, _ = gm
        return group_ids[grid_id], k_i, 0, 0

    return (
        pl.BlockSpec((tm, QBLOCK), lhs_q_map),
        pl.BlockSpec((1, tm, 1), lhs_s_map),
        pl.BlockSpec((None, QBLOCK, n), rhs_q_map),
        pl.BlockSpec((None, 1, 1, n), rhs_s_map),
        n_kblocks,
    )


def _check_args(lhs: ActQuant, rhs: GemmWeight, group_sizes, *, tm: int):
    m, k = lhs.q.shape
    e, kw, n = rhs.q.shape
    if kw != k:
        raise ValueError(f"K mismatch: lhs {k} vs rhs {kw}")
    if k % QBLOCK != 0 or n % QBLOCK != 0:
        raise ValueError(f"(K={k}, N={n}) must be multiples of {QBLOCK}")
    if m % tm != 0:
        raise ValueError(f"M={m} must be a multiple of tm={tm}; pad rows")
    if group_sizes.shape != (e,):
        raise ValueError(f"group_sizes {group_sizes.shape} != ({e},)")
    return m, k, n, e


def gmm_fp8(
    lhs: ActQuant,            # q [M, K], s_t [K/128, M]
    rhs: GemmWeight,          # q [E, K, N], s_bcast [E, K/128, 1, N]
    group_sizes: jax.Array,   # [E] int32, sum <= M
    *,
    tm: int = 128,
    out_dtype=jnp.bfloat16,
    compute_upcast: bool = True,
    interpret: bool = False,
) -> jax.Array:
    """Grouped fp8 GEMM. Rows of ``lhs`` are grouped by expert in order;
    ``out[M, N]`` rows beyond ``sum(group_sizes)`` are unspecified (callers
    gather only routed rows back).
    """
    m, k, n, e = _check_args(lhs, rhs, group_sizes, tm=tm)

    group_metadata, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes.astype(jnp.int32),
        m=m, tm=tm,
        start_group=jnp.zeros((), jnp.int32),
        num_nonzero_groups=e,
        visit_empty_groups=False,
    )
    lhs_q_spec, lhs_s_spec, rhs_q_spec, rhs_s_spec, n_kblocks = _common_specs(
        tm=tm, k=k, n=n)

    out = pl.pallas_call(
        partial(_gmm_kernel, n_ksteps=n_kblocks, tm=tm, n=n,
                compute_upcast=compute_upcast),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_active_tiles, n_kblocks),
            in_specs=[lhs_q_spec, lhs_s_spec, rhs_q_spec, rhs_s_spec],
            out_specs=pl.BlockSpec((tm, n), lambda g, ki, gm: (gm[2][g], 0)),
            scratch_shapes=[pltpu.VMEM((tm, n), jnp.float32)],
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
        ),
        interpret=interpret,
    )(group_metadata, lhs.q, lhs.s_t[..., None], rhs.q, rhs.s_bcast)
    return out


def gmm_fp8_swiglu_quant(
    lhs: ActQuant,            # q [M, K], s_t [K/128, M]
    rhs_w13: GemmWeight,      # q [E, K, 2*dff] (gate ‖ up)
    group_sizes: jax.Array,   # [E]
    *,
    tm: int = 128,
    compute_upcast: bool = True,
    interpret: bool = False,
) -> ActQuant:
    """Fused Linear-1 + SwiGLU + fp8 re-quantization.

    Returns the hidden activation as an ``ActQuant`` (q ``[M, dff]``,
    s_t ``[dff/128, M]``) ready to feed ``gmm_fp8`` for Linear-2. The
    full-precision hidden tensor never touches HBM: per token this saves
    a ``2*dff`` bf16 write + read (vs. unfused) and replaces it with a
    ``dff`` fp8 write — a 8x cut on the intermediate's HBM traffic.
    """
    m, k, n2, e = _check_args(lhs, rhs_w13, group_sizes, tm=tm)
    if n2 % 2 != 0:
        raise ValueError("W13 output dim must be 2*dff (gate ‖ up)")
    dff = n2 // 2
    if dff % QBLOCK != 0:
        raise ValueError(f"dff={dff} must be a multiple of {QBLOCK}")

    group_metadata, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes.astype(jnp.int32),
        m=m, tm=tm,
        start_group=jnp.zeros((), jnp.int32),
        num_nonzero_groups=e,
        visit_empty_groups=False,
    )
    lhs_q_spec, lhs_s_spec, rhs_q_spec, rhs_s_spec, n_kblocks = _common_specs(
        tm=tm, k=k, n=n2)

    h_q, h_s3 = pl.pallas_call(
        partial(_gmm_swiglu_quant_kernel, n_ksteps=n_kblocks, tm=tm, dff=dff,
                compute_upcast=compute_upcast),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_active_tiles, n_kblocks),
            in_specs=[lhs_q_spec, lhs_s_spec, rhs_q_spec, rhs_s_spec],
            out_specs=[
                pl.BlockSpec((tm, dff), lambda g, ki, gm: (gm[2][g], 0)),
                pl.BlockSpec((dff // QBLOCK, tm, 1),
                             lambda g, ki, gm: (0, gm[2][g], 0)),
            ],
            scratch_shapes=[pltpu.VMEM((tm, n2), jnp.float32)],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((m, dff), FP8_DTYPE),
            jax.ShapeDtypeStruct((dff // QBLOCK, m, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
        ),
        interpret=interpret,
    )(group_metadata, lhs.q, lhs.s_t[..., None], rhs_w13.q, rhs_w13.s_bcast)
    return ActQuant(q=h_q, s_t=h_s3[..., 0])


# ---------------------------------------------------------------------------
# Eager references (the oracles tests grade against)
# ---------------------------------------------------------------------------


def gmm_ref(lhs: ActQuant, rhs: WeightQuant, group_sizes: jax.Array) -> jax.Array:
    """fp32 dequantized reference for ``gmm_fp8``."""
    from .quant import dequantize_act, dequantize_weight
    x = dequantize_act(lhs)
    w = dequantize_weight(rhs)
    e = w.shape[0]
    ends = jnp.cumsum(group_sizes)
    starts = ends - group_sizes
    m = x.shape[0]
    rows = jnp.arange(m)
    out = jnp.zeros((m, w.shape[-1]), jnp.float32)
    for g in range(e):
        sel = (rows >= starts[g]) & (rows < ends[g])
        out = out + jnp.where(sel[:, None], x @ w[g], 0.0)
    return out


def swiglu_ref(h13: jax.Array) -> jax.Array:
    dff = h13.shape[-1] // 2
    gate, up = h13[..., :dff], h13[..., dff:]
    return (gate * jax.nn.sigmoid(gate)) * up
