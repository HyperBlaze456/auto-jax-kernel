"""DeepSeekMoE serving layer: routing, expert-parallel dispatch/combine,
wave-pipelined comm/compute overlap.

Pipeline (paper §3.1, Figure 5) and where each stage runs:

    Router (fp32, replicated)                — XLA
    Quantize x once per token (fp8 + scales) — XLA (amortized over top-6)
    Dispatch all-to-all  (FP8 payload)       — ICI, per expert-wave
    Linear-1 + SwiGLU + FP8 cast             — Pallas (gmm_fp8_swiglu_quant)
    Linear-2                                 — Pallas (gmm_fp8)
    Combine all-to-all   (BF16 payload)      — ICI, per expert-wave
    Weighted scatter-add into tokens         — XLA

Comm/compute overlap
--------------------

Experts are partitioned into ``n_waves`` contiguous waves. Each wave's
dispatch a2a → grouped GEMMs → combine a2a forms an *independent
dependency chain*; the chains only join at the final scatter-add. XLA's
latency-hiding scheduler can therefore run wave w+1's dispatch a2a on the
ICI while wave w's GEMMs occupy the MXU — the paper's fine-grained EP
scheme expressed at the dependency-graph level. (A single fused Pallas
mega-kernel with in-kernel ``make_async_remote_copy`` waves is the
endgame; the wave-structured graph already captures the steady-state
overlap with ~zero code risk. See HARDWARE_NOTES.)

ICI accounting per token (Flash config, d=4096, top-6):
  dispatch: 6 × (4096 fp8 + 32×4 scale + 16 meta) ≈ 25.3 KB
  combine:  6 × 4096 × 2 (bf16)                  ≈ 49.2 KB
The FP8 dispatch halves the dispatch volume exactly as the paper's
"3h bytes per token-expert pair (FP8 Dispatch + BF16 Combine)".

Capacity: the a2a needs static shapes, so each (src shard → dst shard,
wave) lane carries ``capacity`` slots. Pairs beyond capacity are dropped
and the surviving gates renormalized (serving deployments size capacity
so this is ~never hit; the eager oracle models the same drop rule so
tests stay exact).

Determinism: the combine scatter-add is a fixed-order ``segment_sum``
over a statically-shaped buffer — no atomics, no order ambiguity (the
TPU analogue of paper §3.3's deterministic MoE combine).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import MoEConfig, ServingTiles
from .gemm_fp8 import GemmWeight, gmm_fp8, gmm_fp8_swiglu_quant
from .quant import ActQuant, quantize_act


class MoEParams(NamedTuple):
    """Kernel-ready MoE parameters (weights pre-quantized at load time)."""

    w_router: jax.Array        # [d, E] (bf16/f32; routing math runs fp32)
    router_bias: jax.Array     # [E] f32 — aux-loss-free selection bias
    w13: GemmWeight            # q [E, d, 2*dff]
    w2: GemmWeight             # q [E, dff, d]
    w13_shared: GemmWeight     # q [1, d, 2*dff_sh]
    w2_shared: GemmWeight      # q [1, dff_sh, d]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route(
    x: jax.Array,              # [M, d]
    params: MoEParams,
    cfg: MoEConfig,
) -> tuple[jax.Array, jax.Array]:
    """Learned routing. Returns ``(idx [M, topk] int32, gates [M, topk] f32)``.

    V4 affinity is ``Sqrt(Softplus(logit))`` (§2.1; changed from V3's
    sigmoid). The aux-loss-free bias enters *selection only*; gates come
    from the unbiased affinities, normalized over the selected set.
    Routing runs in fp32 end-to-end — it is tiny and decides everything.
    """
    logits = x.astype(jnp.float32) @ params.w_router.astype(jnp.float32)
    aff = jnp.sqrt(jax.nn.softplus(logits))                 # [M, E]
    _, idx = jax.lax.top_k(aff + params.router_bias[None, :], cfg.topk)
    gates = jnp.take_along_axis(aff, idx, axis=-1)          # [M, topk]
    if cfg.route_norm_topk:
        gates = gates / (gates.sum(-1, keepdims=True) + 1e-20)
    return idx.astype(jnp.int32), gates


def route_hash(
    token_ids: jax.Array,      # [M] int32 (input token ids)
    cfg: MoEConfig,
) -> tuple[jax.Array, jax.Array]:
    """Hash routing for the first ``n_hash_layers`` MoE layers (§2.1).

    The paper specifies only "a predefined hash function of the input
    token ID"; we use affine probes mod E with equal gates. Deterministic,
    parameter-free, and load-balanced in expectation — the kernel path is
    identical to learned routing.
    """
    j = jnp.arange(cfg.topk, dtype=jnp.uint32)[None, :]
    h = token_ids.astype(jnp.uint32)[:, None] * jnp.uint32(2654435761)  # Knuth
    idx = ((h + j * jnp.uint32(40503)) % jnp.uint32(cfg.n_routed)).astype(jnp.int32)
    gates = jnp.full((token_ids.shape[0], cfg.topk),
                     1.0 / cfg.topk, jnp.float32)
    return idx, gates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_rows(m: int, tm: int) -> int:
    return ((m + tm - 1) // tm) * tm


def _gather_act(aq: ActQuant, rows: jax.Array, m_pad: int) -> ActQuant:
    """Gather token rows of a quantized activation (payload + scale cols),
    zero-padded to ``m_pad`` rows."""
    q = jnp.zeros((m_pad, aq.q.shape[1]), aq.q.dtype).at[
        : rows.shape[0]].set(aq.q[rows])
    s_t = jnp.zeros((aq.s_t.shape[0], m_pad), aq.s_t.dtype).at[
        :, : rows.shape[0]].set(aq.s_t[:, rows])
    return ActQuant(q=q, s_t=s_t)


def _expert_gemms(
    x_sorted: ActQuant,
    group_sizes: jax.Array,
    w13: GemmWeight,
    w2: GemmWeight,
    *,
    tiles: ServingTiles,
    compute_upcast: bool,
) -> jax.Array:
    """Linear-1 + SwiGLU + fp8 cast + Linear-2 on expert-sorted rows."""
    h = gmm_fp8_swiglu_quant(
        x_sorted, w13, group_sizes,
        tm=tiles.gemm_tm, compute_upcast=compute_upcast,
        interpret=tiles.interpret)
    return gmm_fp8(
        h, w2, group_sizes,
        tm=tiles.gemm_tm, out_dtype=jnp.bfloat16,
        compute_upcast=compute_upcast, interpret=tiles.interpret)


def shared_expert(
    xq: ActQuant,              # [M_pad, d] — all rows valid
    params: MoEParams,
    *,
    tiles: ServingTiles,
    compute_upcast: bool = True,
) -> jax.Array:
    """Always-on shared expert: a 1-group ``gmm`` (E=1), same kernels."""
    m = xq.q.shape[0]
    return _expert_gemms(
        xq, jnp.array([m], jnp.int32), params.w13_shared, params.w2_shared,
        tiles=tiles, compute_upcast=compute_upcast)


# ---------------------------------------------------------------------------
# Single-shard MoE (EP=1 fast path; also the per-shard compute core)
# ---------------------------------------------------------------------------


def moe_forward_local(
    x: jax.Array,              # [M, d] bf16/f32
    idx: jax.Array,            # [M, topk] int32 routed experts
    gates: jax.Array,          # [M, topk] f32
    params: MoEParams,
    cfg: MoEConfig,
    *,
    tiles: ServingTiles,
    compute_upcast: bool | None = None,
) -> jax.Array:
    """All experts resident on this device. Tokens are quantized **once**
    (the fp8 payload is reused by all top-k replicas and the shared
    expert), sorted by expert, pushed through the fused expert GEMMs, and
    combined with a fixed-order segment-sum."""
    if compute_upcast is None:
        compute_upcast = tiles.compute_upcast
    m, d = x.shape
    e, topk = cfg.n_routed, cfg.topk
    p = m * topk

    xq = quantize_act(x.reshape(m, d))

    pair_expert = idx.reshape(p)
    pair_src = jnp.repeat(jnp.arange(m, dtype=jnp.int32), topk)
    pair_gate = gates.reshape(p).astype(jnp.float32)

    order = jnp.argsort(pair_expert)                       # stable
    group_sizes = jnp.bincount(pair_expert, length=e).astype(jnp.int32)

    m_pad = _pad_rows(p, tiles.gemm_tm)
    x_sorted = _gather_act(xq, pair_src[order], m_pad)

    y = _expert_gemms(x_sorted, group_sizes, params.w13, params.w2,
                      tiles=tiles, compute_upcast=compute_upcast)  # [m_pad, d]

    # Combine: weighted scatter-add back to tokens, fixed order.
    y = y[:p].astype(jnp.float32) * pair_gate[order][:, None]
    out = jax.ops.segment_sum(y, pair_src[order], num_segments=m)

    # Shared expert reuses the same quantized payload.
    m_pad_tok = _pad_rows(m, tiles.gemm_tm)
    x_tok = _gather_act(xq, jnp.arange(m, dtype=jnp.int32), m_pad_tok)
    out = out + shared_expert(x_tok, params, tiles=tiles,
                              compute_upcast=compute_upcast)[:m].astype(jnp.float32)
    return out.astype(x.dtype)


# ---------------------------------------------------------------------------
# Expert-parallel MoE (shard_map body) with wave overlap
# ---------------------------------------------------------------------------


def moe_forward_ep(
    x: jax.Array,              # [M_local, d] — this shard's tokens
    idx: jax.Array,            # [M_local, topk]
    gates: jax.Array,          # [M_local, topk]
    params: MoEParams,         # expert-sharded leaves: w13/w2 are [E_local,...]
    cfg: MoEConfig,
    *,
    axis_name: str,
    ep_size: int,
    n_waves: int,
    capacity: int,             # slots per (src, dst, wave) lane
    tiles: ServingTiles,
    compute_upcast: bool | None = None,
) -> jax.Array:
    """Runs *inside* ``shard_map`` over ``axis_name``. Expert weights are
    sharded ``E_local = E / ep_size`` per device; tokens stay with their
    shard. Each wave: bucket pairs by destination shard → FP8 dispatch
    a2a → per-shard grouped GEMMs → BF16 combine a2a → scatter-add.

    Waves are independent dependency chains; their a2a's overlap with
    neighbouring waves' GEMMs under XLA's latency-hiding scheduler.
    """
    if compute_upcast is None:
        compute_upcast = tiles.compute_upcast
    m, d = x.shape
    e, topk = cfg.n_routed, cfg.topk
    e_local = e // ep_size
    e_wave = e // n_waves
    if e % (ep_size * n_waves) or e_local % n_waves:
        raise ValueError("n_routed must divide evenly by ep_size and n_waves")
    p = m * topk
    my_shard = jax.lax.axis_index(axis_name)

    xq = quantize_act(x.reshape(m, d))
    nsk = xq.s_t.shape[0]                                  # d / 128

    pair_expert = idx.reshape(p)
    pair_src = jnp.repeat(jnp.arange(m, dtype=jnp.int32), topk)
    pair_gate = gates.reshape(p).astype(jnp.float32)

    out = jnp.zeros((m, d), jnp.float32)
    for w in range(n_waves):
        # ---- pairs of this wave ----
        in_wave = (pair_expert >= w * e_wave) & (pair_expert < (w + 1) * e_wave)
        dest = jnp.where(in_wave, pair_expert // e_local, ep_size)  # ep = drop
        sort_ix = jnp.argsort(dest)                         # stable
        dest_s = dest[sort_ix]
        # position within destination bucket
        dcount = jnp.bincount(dest, length=ep_size + 1)
        dstart = jnp.concatenate([jnp.zeros(1, jnp.int32),
                                  jnp.cumsum(dcount)[:-1].astype(jnp.int32)])
        pos = jnp.arange(p, dtype=jnp.int32) - dstart[dest_s]
        keep = (dest_s < ep_size) & (pos < capacity)

        # ---- pack send buffers [ep, capacity, ...] ----
        slot = jnp.where(keep, dest_s * capacity + pos, ep_size * capacity)
        pair_ix = sort_ix                                   # pair id per sorted row

        def scatter(payload, fill, shape_tail, dtype):
            buf = jnp.full((ep_size * capacity + 1,) + shape_tail, fill, dtype)
            return buf.at[slot].set(payload)[:-1].reshape(
                (ep_size, capacity) + shape_tail)

        send_q = scatter(xq.q[pair_src[pair_ix]], 0, (d,), xq.q.dtype)
        send_s = scatter(xq.s_t[:, pair_src[pair_ix]].T, 0.0, (nsk,), jnp.float32)
        send_e = scatter(pair_expert[pair_ix], -1, (), jnp.int32)
        send_r = scatter(pair_ix, -1, (), jnp.int32)        # pair id (for return)

        # ---- dispatch a2a (FP8 payload + scales + meta) ----
        a2a = partial(jax.lax.all_to_all, axis_name=axis_name,
                      split_axis=0, concat_axis=0)
        recv_q, recv_s, recv_e = a2a(send_q), a2a(send_s), a2a(send_e)

        # ---- local grouped GEMMs over received rows ----
        rq = recv_q.reshape(ep_size * capacity, d)
        rs = recv_s.reshape(ep_size * capacity, nsk).T
        re = recv_e.reshape(ep_size * capacity)
        # local expert id; invalid rows -> sentinel group e_local (sorted last)
        rel = jnp.where(re >= 0, re - my_shard * e_local, e_local)
        rorder = jnp.argsort(rel)
        m_pad = _pad_rows(ep_size * capacity, tiles.gemm_tm)
        x_sorted = _gather_act(ActQuant(q=rq, s_t=rs), rorder, m_pad)
        group_sizes = jnp.bincount(rel, length=e_local + 1)[:e_local].astype(jnp.int32)

        # All e_local weight tensors are passed, but only wave-active
        # experts have nonzero groups — the grouped GEMM's metadata visits
        # only those, so off-wave expert weights are never fetched.
        y = _expert_gemms(x_sorted, group_sizes, params.w13, params.w2,
                          tiles=tiles, compute_upcast=compute_upcast)

        # un-sort back to recv-buffer order; zero invalid rows
        y_recv = jnp.zeros((ep_size * capacity, d), y.dtype).at[rorder].set(
            y[: ep_size * capacity])
        y_recv = jnp.where((re >= 0)[:, None], y_recv, 0).reshape(
            ep_size, capacity, d)

        # ---- combine a2a (BF16) ----
        y_back = a2a(y_recv.astype(jnp.bfloat16))           # [ep, capacity, d]
        y_back = y_back.reshape(ep_size * capacity, d)
        ids_back = send_r.reshape(ep_size * capacity)       # pair ids we sent

        valid = ids_back >= 0
        pair_safe = jnp.where(valid, ids_back, 0)
        contrib = (y_back.astype(jnp.float32)
                   * pair_gate[pair_safe][:, None]
                   * valid[:, None])
        out = out + jax.ops.segment_sum(
            contrib, jnp.where(valid, pair_src[pair_safe], m),
            num_segments=m + 1)[:m]

    # ---- shared expert (replicated weights), overlaps with nothing to
    # lose: it has no comm dependency at all ----
    m_pad_tok = _pad_rows(m, tiles.gemm_tm)
    x_tok = _gather_act(xq, jnp.arange(m, dtype=jnp.int32), m_pad_tok)
    out = out + shared_expert(x_tok, params, tiles=tiles,
                              compute_upcast=compute_upcast)[:m].astype(jnp.float32)
    return out.astype(x.dtype)


# ---------------------------------------------------------------------------
# Eager oracle
# ---------------------------------------------------------------------------


def moe_ref(
    x: jax.Array,              # [M, d]
    idx: jax.Array,            # [M, topk]
    gates: jax.Array,          # [M, topk]
    params: MoEParams,
    cfg: MoEConfig,
) -> jax.Array:
    """Dense fp32 reference over *dequantized* weights and the same
    once-per-token activation quantization. Grades the kernel path down
    to fp32 reassociation + the (inherent) fp8 hidden re-quantization."""
    from .gemm_fp8 import swiglu_ref
    from .quant import WeightQuant, dequantize_act, dequantize_weight, quantize_act

    def undo_bcast(gw: GemmWeight) -> jax.Array:
        s = gw.s_bcast[..., 0, ::128]                       # [..., K/128, N/128]
        return dequantize_weight(WeightQuant(q=gw.q, s=s))

    m, d = x.shape
    xd = dequantize_act(quantize_act(x.reshape(m, d)))      # model the fp8 input
    w13 = undo_bcast(params.w13)                            # [E, d, 2dff] f32
    w2 = undo_bcast(params.w2)

    out = jnp.zeros((m, d), jnp.float32)
    for j in range(cfg.topk):
        ej = idx[:, j]
        h13 = jnp.einsum("md,mdf->mf", xd, w13[ej])
        h = dequantize_act(quantize_act(swiglu_ref(h13)))   # model fp8 hidden
        out = out + gates[:, j:j+1] * jnp.einsum("mf,mfd->md", h, w2[ej])

    w13s = undo_bcast(params.w13_shared)[0]
    w2s = undo_bcast(params.w2_shared)[0]
    hs = dequantize_act(quantize_act(swiglu_ref(xd @ w13s)))
    out = out + hs @ w2s
    return out.astype(x.dtype)
