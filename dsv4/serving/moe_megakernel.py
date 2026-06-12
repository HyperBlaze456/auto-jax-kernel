"""EP mega-kernel: fused dispatch / expert-GEMM / combine wave pipeline.

This is the MegaMoE idea (paper §3.1, Fig. 5c) in TPU form: ONE Pallas
kernel per shard owns the whole MoE layer — remote-DMA dispatch of wave
w+1 runs on the DMA engines while wave w's expert GEMMs run on the MXU,
and each expert's results are sent back the moment that expert finishes
("result sending of completed experts proceeds concurrently"). The XLA
wave-graph path (``moe.moe_forward_ep``) relies on the scheduler to find
this overlap across op boundaries; here the overlap is *structural* —
encoded in the kernel's instruction order, with no per-wave dispatch
overhead or a2a buffer materialization between stages.

Layout: expert-major static capacity buckets
--------------------------------------------

The single hardest part of grouped expert compute is that group sizes are
data-dependent. GPU MegaMoE solves it with dynamic scheduling across SMs;
the TPU answer is to make the problem *static*: the host-side prep
(``pack_dispatch``, plain XLA) buckets each (wave, dst-shard, local-
expert) lane to a fixed ``cap_e`` slots, so the kernel sees

    send_q [n_waves, ep, e_wl, cap_e, d]   fp8   (+ scales, + pair ids)

and every GEMM tile, every DMA extent, every BlockSpec index is a static
function of grid coordinates. No in-kernel sort, no group metadata, no
scalar prefetch. The price is zero-padded slots (wasted MXU flops at low
load factor) — the standard TPU capacity trade, and exactly what makes
the remote copies fixed-extent so they can be issued before the data is
inspected.

Pipeline structure (grid = (n_waves, e_wl), both sequential)
------------------------------------------------------------

    (w, 0):  start dispatch remote-DMAs for wave w+1   ── DMA engines
             wait dispatch recvs for wave w            ── (already in
                                                          flight since
                                                          step w-1)
    (w, e):  GEMM1+SwiGLU+GEMM2 for local expert e     ── MXU
             start combine remote-DMA of e's results   ── DMA engines
    (last):  wait all combine recvs

Steady state: wave w's compute hides wave w+1's dispatch AND wave w-1's
combine — the paper's three-way concurrency, expressed as semaphore
ordering inside one kernel.

ICI traffic is identical to the wave-graph path (fp8 dispatch + bf16
combine; the bytes are already minimal) — what the fusion buys is the
*continuity* of that traffic: DMA engines never wait for XLA op
boundaries, which is where the paper's 1.4→1.9x headroom lives.

Semaphore accounting (SPMD-symmetric, per kernel_refs §F):
  - ``send_sem[ep]`` / ``recv_sem[ep]`` (dispatch), same pair for
    combine. My ``recv_sem[s]`` is signaled by shard s's DMA into my
    buffers; symmetric code means every wait has a matching remote start.
  - Cross-grid-step persistence: sems live in scratch, started at step
    (w, ·) and waited at (w+1, ·).
  - A collective barrier brackets the kernel (``get_barrier_semaphore``)
    so no shard's sends race a neighbor still in its previous layer.

VMEM: the third grid axis steps both GEMM contractions in 128-row quant
blocks (phase 1: k < d/128 over W13; phase 2: over W2), so per-step VMEM
holds one (QBLOCK, N) weight tile (~0.75 MiB fp8 at Pro shapes), never a
whole expert — Pro's 42 MiB/expert W13 streams through any generation's
VMEM. The in-kernel math is gmm_fp8's two-level accumulation verbatim
(bf16-upcast fp8 dot → f32 partial → × row·col scales per quant block →
f32 master accumulator), with the SwiGLU + fp8 hidden re-quant executed
in-register at the phase boundary. Total resident scratch at Pro shapes
(rows=64): accumulators + fp8 x/h + staging ≈ 5 MiB.

The fallback ``mega_moe_reference`` runs the identical bucket layout
through plain jnp — it is both the correctness oracle and the
shape-contract documentation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import MoEConfig, QBLOCK
from .quant import quantize_act


class DispatchBuckets(NamedTuple):
    """Expert-major, capacity-padded dispatch buffers (one shard's sends).

    Shapes (local view inside shard_map; E_wl = experts per shard per wave):
      q        [n_waves, ep, e_wl, cap_e, d]    e4m3  payload
      s        [n_waves, ep, e_wl, cap_e, nsk]  f32   1x128 scales
      pair_id  [n_waves, ep, e_wl, cap_e]       int32 sender-side pair id, -1 empty
      gate     [P]                              f32   per-pair combine weight
      src_tok  [P]                              int32 per-pair source token row
      n_drop   []                               int32 pairs dropped (capacity)
    """

    q: jax.Array
    s: jax.Array
    pair_id: jax.Array
    gate: jax.Array
    src_tok: jax.Array
    n_drop: jax.Array


def pack_dispatch(
    x: jax.Array,              # [M, d] this shard's tokens
    idx: jax.Array,            # [M, topk] global expert ids
    gates: jax.Array,          # [M, topk]
    cfg: MoEConfig,
    *,
    ep_size: int,
    n_waves: int,
    cap_e: int,
) -> DispatchBuckets:
    """Bucket (token, expert) pairs into the static expert-major layout.

    Slot assignment must be *deterministic*: pairs are ranked by pair id
    within each (wave, dst, expert) lane via a stable argsort, so the
    packed buffers — and therefore everything downstream including the
    remote DMA payloads — are bit-reproducible. Pairs beyond ``cap_e``
    are dropped (counted in ``n_drop``; size capacity so this is ~never).
    """
    m, d = x.shape
    e, topk = cfg.n_routed, cfg.topk
    e_local = e // ep_size
    e_wl = e_local // n_waves
    p = m * topk
    nsk = d // QBLOCK

    xq = quantize_act(x)

    pair_expert = idx.reshape(p)                      # global expert id
    pair_src = jnp.repeat(jnp.arange(m, dtype=jnp.int32), topk)
    pair_gate = gates.reshape(p).astype(jnp.float32)

    dst = pair_expert // e_local                      # owning shard
    e_loc = pair_expert % e_local                     # local expert on dst
    wave = e_loc // e_wl
    e_in_wave = e_loc % e_wl

    # lane = flat (wave, dst, e_in_wave) bucket; rank pairs within lane.
    lane = (wave * ep_size + dst) * e_wl + e_in_wave  # [P]
    n_lanes = n_waves * ep_size * e_wl
    order = jnp.argsort(lane)                          # stable → by pair id
    lane_sorted = lane[order]
    lane_start = jnp.searchsorted(lane_sorted, jnp.arange(n_lanes))
    rank = jnp.arange(p, dtype=jnp.int32) - lane_start[lane_sorted]

    keep = rank < cap_e
    slot = jnp.where(keep, lane_sorted * cap_e + rank, n_lanes * cap_e)

    def scatter(payload_sorted, fill, tail, dtype):
        buf = jnp.full((n_lanes * cap_e + 1,) + tail, fill, dtype)
        buf = buf.at[slot].set(payload_sorted)
        return buf[:-1].reshape((n_waves, ep_size, e_wl, cap_e) + tail)

    pid = order.astype(jnp.int32)                      # pair id per sorted row
    q_b = scatter(xq.q[pair_src[order]], 0, (d,), xq.q.dtype)
    s_b = scatter(xq.s_t[:, pair_src[order]].T, 0.0, (nsk,), jnp.float32)
    id_b = scatter(pid, -1, (), jnp.int32)
    n_drop = (~keep).sum().astype(jnp.int32)

    return DispatchBuckets(q=q_b, s=s_b, pair_id=id_b,
                           gate=pair_gate, src_tok=pair_src, n_drop=n_drop)


def combine_results(
    y_back: jax.Array,         # [n_waves, ep, e_wl, cap_e, d] results (my pairs)
    ids_back: jax.Array,       # [n_waves, ep, e_wl, cap_e] my pair ids
    buckets: DispatchBuckets,
    m: int,
) -> jax.Array:
    """Deterministic weighted scatter-add of returned expert outputs back
    to token rows (fixed-order segment-sum, as everywhere else)."""
    d = y_back.shape[-1]
    yb = y_back.reshape(-1, d).astype(jnp.float32)
    ib = ids_back.reshape(-1)
    valid = ib >= 0
    safe = jnp.where(valid, ib, 0)
    contrib = yb * buckets.gate[safe][:, None] * valid[:, None]
    seg = jnp.where(valid, buckets.src_tok[safe], m)
    return jax.ops.segment_sum(contrib, seg, num_segments=m + 1)[:m]


# ---------------------------------------------------------------------------
# Bucket-layout reference (oracle + shape contract; plain jnp, no comms)
# ---------------------------------------------------------------------------


def _expert_ffn_dense(xq_rows: jax.Array, s_rows: jax.Array,
                      w13_e: jax.Array, w2_e: jax.Array) -> jax.Array:
    """One expert over its [rows, d] fp8 bucket — fp32 two-level accum,
    numerically the same recipe as gmm_fp8 (dequant-scale per 128-block
    folds into one row/col scale product here because rows share scales
    per k-block)."""
    d = xq_rows.shape[-1]
    nsk = d // QBLOCK
    x_deq = (xq_rows.astype(jnp.float32).reshape(-1, nsk, QBLOCK)
             * s_rows[..., None]).reshape(-1, d)
    h13 = x_deq @ w13_e                                   # [rows, 2dff] f32
    dff = h13.shape[-1] // 2
    h = (h13[:, :dff] * jax.nn.sigmoid(h13[:, :dff])) * h13[:, dff:]
    hq = quantize_act(h)
    h_deq = (hq.q.astype(jnp.float32).reshape(-1, dff // QBLOCK, QBLOCK)
             * hq.s_t.T[..., None]).reshape(-1, dff)
    return h_deq @ w2_e                                   # [rows, d] f32


def mega_moe_reference(
    buckets: DispatchBuckets,
    w13_deq: jax.Array,        # [E_local, d, 2dff] f32 (this shard's experts,
    w2_deq: jax.Array,         #  dequantized) — single-shard reference: the
    m: int,                    #  "all-to-all" is an identity (ep_size=1)
) -> jax.Array:
    """Single-shard oracle over the exact bucket layout the kernel sees.
    With ep_size=1 dispatch/combine are identities, so this isolates the
    bucket bookkeeping + expert math from the comms."""
    n_waves, ep, e_wl, cap_e, d = buckets.q.shape
    assert ep == 1, "reference models the ep_size=1 (no-comms) case"
    out_rows = jnp.zeros((n_waves, ep, e_wl, cap_e, d), jnp.float32)
    for w in range(n_waves):
        for e in range(e_wl):
            le = w * e_wl + e
            y = _expert_ffn_dense(
                buckets.q[w, 0, e], buckets.s[w, 0, e],
                w13_deq[le], w2_deq[le])
            out_rows = out_rows.at[w, 0, e].set(y)
    return combine_results(out_rows, buckets.pair_id, buckets, m)


# ---------------------------------------------------------------------------
# The mega-kernel proper
# ---------------------------------------------------------------------------
#
# One pallas_call per shard inside shard_map('ep'). Grid (n_waves, e_wl),
# both sequential. Remote-DMA endpoints are ANY-space HBM refs; semaphore
# arrays are parity-indexed over waves (only waves w and w+1 are ever in
# flight, so (2, ep) slots disambiguate same-shape descriptors that would
# otherwise race on a shared semaphore's byte count).
#
# Interpret-mode note: remote DMA requires interpret=pltpu.InterpretParams()
# (the Mosaic interpreter), NOT interpret=True — the HLO-discharge path
# crashes on tuple device_ids and multi-descriptor waits in jax 0.10.1.

from functools import partial  # noqa: E402

import jax.experimental.pallas as pl  # noqa: E402
import jax.experimental.pallas.tpu as pltpu  # noqa: E402

from .config import FP8_MAX  # noqa: E402
from .gemm_fp8 import GemmWeight  # noqa: E402

_EPSQ = 1e-12


def _mega_kernel(
    # inputs
    send_q_hbm,    # ANY [n_waves, ep, e_wl, cap_e, d]    e4m3  my dispatch payloads
    send_s_hbm,    # ANY [n_waves, ep, e_wl, cap_e, nsk]  f32
    w13_q_ref,     # [1, QBLOCK, 2dff] e4m3  (k-th quant-block row tile of
                   #  this (wave, e)'s expert W13, streamed via BlockSpec)
    w13_s_ref,     # [1, 1, 1, 2dff]  f32    (lane-broadcast scales, k-th block)
    w2_q_ref,      # [1, QBLOCK, d]   e4m3   (W2's (k - nk1)-th block tile)
    w2_s_ref,      # [1, 1, 1, d]     f32
    # outputs
    recv_q_hbm,    # ANY [n_waves, ep, e_wl, cap_e, d]    e4m3  dispatch landing
    recv_s_hbm,    # ANY [n_waves, ep, e_wl, cap_e, nsk]  f32
    stage_hbm,     # ANY [n_waves, ep, e_wl, cap_e, d]    bf16  my computed results
    y_back_hbm,    # ANY [n_waves, ep, e_wl, cap_e, d]    bf16  combine landing
    # scratch
    xq_vmem,       # [ep * cap_e, d]    e4m3
    xs_vmem,       # [ep * cap_e, nsk]  f32
    hq_vmem,       # [ep * cap_e, dff]  e4m3  (fp8 hidden between the GEMMs)
    hs_vmem,       # [ep * cap_e, nhk]  f32
    acc1_vmem,     # [ep * cap_e, 2dff] f32   GEMM1 accumulator
    acc2_vmem,     # [ep * cap_e, d]    f32   GEMM2 accumulator
    y_vmem,        # [ep * cap_e, d]    bf16
    disp_send_sem,  # DMA (2, ep)
    disp_recv_sem,  # DMA (2, ep)
    comb_send_sem,  # DMA (ep,)
    comb_recv_sem,  # DMA (ep,)
    load_sem,       # DMA
    *,
    axis_name: str,
    n_waves: int,
    ep: int,
    e_wl: int,
    cap_e: int,
    d: int,
    nsk: int,
    dff: int,
):
    w = pl.program_id(0)
    e = pl.program_id(1)
    k = pl.program_id(2)
    nk1 = nsk                 # GEMM1 contraction steps (d / QBLOCK)
    nhk = dff // QBLOCK       # GEMM2 contraction steps (dff / QBLOCK)
    my_id = jax.lax.axis_index(axis_name)

    def disp_ops(wv, *, for_start: bool):
        """Dispatch descriptors for wave wv: my send_q[wv, j] → shard j's
        recv_q[wv, my_id], parity-sliced semaphores.

        Semaphore slot convention (the part that deadlocks if you get it
        wrong): the recv_sem named in a START descriptor is an *address
        resolved on the destination* — so the sender must name slot
        ``my_id`` (its own id), which lands on the destination's
        ``recv_sem[sender]``. The WAIT descriptors run locally and name
        slot ``j`` = "data from shard j has arrived". Start and wait
        therefore use different recv slots; the send slot (my send to j,
        waited locally) is ``j`` in both."""
        par = jax.lax.rem(wv, 2)
        ops = []
        for j in range(ep):
            recv_slot = my_id if for_start else j
            for src_hbm, dst_hbm in ((send_q_hbm, recv_q_hbm),
                                     (send_s_hbm, recv_s_hbm)):
                ops.append(pltpu.make_async_remote_copy(
                    src_hbm.at[wv, j],
                    dst_hbm.at[wv, my_id],
                    disp_send_sem.at[par, j],
                    disp_recv_sem.at[par, recv_slot],
                    device_id=(j,),
                    device_id_type=pl.DeviceIdType.MESH,
                ))
        return ops

    # ---- wave-pipelined dispatch: send w+1 while computing w ----
    @pl.when(jnp.logical_and(e == 0, k == 0))
    def _dispatch():
        @pl.when(w == 0)
        def _warmup():
            for op in disp_ops(0, for_start=True):
                op.start()

        @pl.when(w + 1 < n_waves)
        def _next_wave():
            for op in disp_ops(w + 1, for_start=True):
                op.start()

        # Wait wave w's landings (each shard's send into my recv buffers)
        # AND my own wave-w sends (so their source slabs are reusable).
        for op in disp_ops(w, for_start=False):
            op.wait()

    # ---- load this expert's received rows once per (w, e) ----
    @pl.when(k == 0)
    def _load_rows():
        loads = []
        for src in range(ep):
            loads.append(pltpu.make_async_copy(
                recv_q_hbm.at[w, src, e],
                xq_vmem.at[pl.ds(src * cap_e, cap_e)], load_sem))
            loads.append(pltpu.make_async_copy(
                recv_s_hbm.at[w, src, e],
                xs_vmem.at[pl.ds(src * cap_e, cap_e)], load_sem))
        for op in loads:
            op.start()
        for op in loads:
            op.wait()
        acc1_vmem[...] = jnp.zeros_like(acc1_vmem)
        acc2_vmem[...] = jnp.zeros_like(acc2_vmem)

    rows = ep * cap_e

    # ---- phase 1 (k < nk1): GEMM1 over W13's k-th 128-row quant block.
    # Exactly gmm_fp8's two-level accumulation: bf16-upcast fp8 dot →
    # fp32 partial → × (row scale · lane-broadcast col scale) → master acc.
    @pl.when(k < nk1)
    def _gemm1_step():
        xq = jax.lax.dynamic_slice(
            xq_vmem[...], (0, k * QBLOCK), (rows, QBLOCK))
        xs = jax.lax.dynamic_slice(xs_vmem[...], (0, k), (rows, 1))
        # None-squeezed BlockSpec leading dims: refs are already [128, N].
        part = jax.lax.dot_general(
            xq.astype(jnp.bfloat16), w13_q_ref[...].astype(jnp.bfloat16),
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32)            # [rows, 2dff]
        acc1_vmem[...] += part * xs * w13_s_ref[...].reshape(1, 2 * dff)

    # ---- phase boundary: SwiGLU + fp8 re-quant of the hidden, persisted
    # to scratch so phase 2's steps slice it block by block.
    @pl.when(k == nk1 - 1)
    def _swiglu_quant():
        acc = acc1_vmem[...]
        g, u = acc[:, :dff], acc[:, dff:]
        h = (g * jax.nn.sigmoid(g)) * u                    # f32 [rows, dff]
        h3 = h.reshape(rows, nhk, QBLOCK)
        hs = jnp.maximum(jnp.max(jnp.abs(h3), axis=-1), _EPSQ) / FP8_MAX
        hq = jnp.clip(h3 / hs[..., None], -FP8_MAX, FP8_MAX)
        hq_vmem[...] = hq.reshape(rows, dff).astype(jnp.float8_e4m3fn)
        hs_vmem[...] = hs

    # ---- phase 2 (k >= nk1): GEMM2 over W2's (k - nk1)-th quant block.
    @pl.when(k >= nk1)
    def _gemm2_step():
        k2 = k - nk1
        hq = jax.lax.dynamic_slice(
            hq_vmem[...], (0, k2 * QBLOCK), (rows, QBLOCK))
        hs = jax.lax.dynamic_slice(hs_vmem[...], (0, k2), (rows, 1))
        part = jax.lax.dot_general(
            hq.astype(jnp.bfloat16), w2_q_ref[...].astype(jnp.bfloat16),
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32)            # [rows, d]
        acc2_vmem[...] += part * hs * w2_s_ref[...].reshape(1, d)

    # ---- finalize (last k): stage + per-expert eager combine ----
    @pl.when(k == nk1 + nhk - 1)
    def _finalize():
        # ANY-space refs can't be stored to directly — stage via VMEM + DMA.
        y_vmem[...] = acc2_vmem[...].astype(jnp.bfloat16)
        stores = [
            pltpu.make_async_copy(
                y_vmem.at[pl.ds(r * cap_e, cap_e)],
                stage_hbm.at[w, r, e], load_sem)
            for r in range(ep)
        ]
        for op in stores:
            op.start()
        for op in stores:
            op.wait()

        # Same slot convention as dispatch: starts signal the destination's
        # comb_recv_sem[my_id]; the drain waits slot r per source shard.
        for r in range(ep):
            pltpu.make_async_remote_copy(
                stage_hbm.at[w, r, e],
                y_back_hbm.at[w, my_id, e],
                comb_send_sem.at[r],
                comb_recv_sem.at[my_id],
                device_id=(r,),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()

    # ---- drain: at the last grid step, absorb every combine descriptor.
    # Same-shape slabs make wait-by-descriptor order-insensitive; the
    # semaphore byte counts total exactly n_waves*e_wl*ep descriptors.
    @pl.when(jnp.logical_and(
        jnp.logical_and(w == n_waves - 1, e == e_wl - 1),
        k == nk1 + nhk - 1))
    def _drain():
        for wv in range(n_waves):
            for ee in range(e_wl):
                for r in range(ep):
                    pltpu.make_async_remote_copy(
                        stage_hbm.at[wv, r, ee],
                        y_back_hbm.at[wv, my_id, ee],
                        comb_send_sem.at[r],
                        comb_recv_sem.at[r],
                        device_id=(r,),
                        device_id_type=pl.DeviceIdType.MESH,
                    ).wait()


def mega_moe_shard(
    buckets: DispatchBuckets,
    w13: "GemmWeight",         # q [E_local, d, 2dff] e4m3, s_bcast [E_local, d/128, 1, 2dff]
    w2: "GemmWeight",          # q [E_local, dff, d] e4m3, s_bcast [E_local, dff/128, 1, d]
    m: int,
    *,
    axis_name: str,
    ep_size: int,
    n_waves: int,
) -> jax.Array:
    """Per-shard body (call inside shard_map): mega-kernel + combine.
    Returns this shard's routed-expert output [m, d] f32 (caller adds the
    shared expert; gates are folded by combine_results).

    Weights stream as 128-row quant-block tiles through the third grid
    axis — per-step VMEM is one (QBLOCK, N) tile per weight, not the whole
    expert, so Pro shapes (42 MiB/expert W13) fit any generation's VMEM.
    The clamped index_maps pin the off-phase weight to its last/first
    block, so Pallas's revisit rule fetches nothing extra during the
    other phase."""
    n_waves_b, ep, e_wl, cap_e, d = buckets.q.shape
    assert (n_waves_b, ep) == (n_waves, ep_size)
    nsk = buckets.s.shape[-1]
    e_local, dff, _ = w2.q.shape
    assert e_local == n_waves * e_wl
    nk1 = d // QBLOCK
    nhk = dff // QBLOCK

    grid = (n_waves, e_wl, nk1 + nhk)
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    buf_shape = jax.ShapeDtypeStruct(buckets.q.shape, buckets.q.dtype)
    sbuf_shape = jax.ShapeDtypeStruct(buckets.s.shape, jnp.float32)
    y_shape = jax.ShapeDtypeStruct((n_waves, ep, e_wl, cap_e, d), jnp.bfloat16)

    def le(w, e):
        return w * e_wl + e

    _, _, _, y_back = pl.pallas_call(
        partial(_mega_kernel, axis_name=axis_name, n_waves=n_waves, ep=ep,
                e_wl=e_wl, cap_e=cap_e, d=d, nsk=nsk, dff=dff),
        grid=grid,
        in_specs=[
            any_spec, any_spec,
            pl.BlockSpec((None, QBLOCK, 2 * dff),
                         lambda w, e, k: (le(w, e), jnp.minimum(k, nk1 - 1), 0)),
            pl.BlockSpec((None, 1, 1, 2 * dff),
                         lambda w, e, k: (le(w, e), jnp.minimum(k, nk1 - 1), 0, 0)),
            pl.BlockSpec((None, QBLOCK, d),
                         lambda w, e, k: (le(w, e),
                                          jnp.clip(k - nk1, 0, nhk - 1), 0)),
            pl.BlockSpec((None, 1, 1, d),
                         lambda w, e, k: (le(w, e),
                                          jnp.clip(k - nk1, 0, nhk - 1), 0, 0)),
        ],
        out_specs=[any_spec, any_spec, any_spec, any_spec],
        out_shape=[buf_shape, sbuf_shape, y_shape, y_shape],
        scratch_shapes=[
            pltpu.VMEM((ep * cap_e, d), buckets.q.dtype),
            pltpu.VMEM((ep * cap_e, nsk), jnp.float32),
            pltpu.VMEM((ep * cap_e, dff), jnp.float8_e4m3fn),
            pltpu.VMEM((ep * cap_e, nhk), jnp.float32),
            pltpu.VMEM((ep * cap_e, 2 * dff), jnp.float32),
            pltpu.VMEM((ep * cap_e, d), jnp.float32),
            pltpu.VMEM((ep * cap_e, d), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((2, ep)),
            pltpu.SemaphoreType.DMA((2, ep)),
            pltpu.SemaphoreType.DMA((ep,)),
            pltpu.SemaphoreType.DMA((ep,)),
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
            collective_id=0,
        ),
        interpret=pltpu.InterpretParams(),
    )(buckets.q, buckets.s, w13.q, w13.s_bcast, w2.q, w2.s_bcast)

    # y_back[w, dst, e, c] is the result of MY send slot [w, dst, e, c] —
    # pair ids never travel over the wire.
    return combine_results(y_back, buckets.pair_id, buckets, m)
