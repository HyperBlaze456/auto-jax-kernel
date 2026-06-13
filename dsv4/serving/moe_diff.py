"""Differentiable local MoE layer for training.

Composition principle: only the expert FFN is a custom_vjp unit
(``grouped_ffn`` — fp8 forward, bf16 backward, fp8-residual recompute).
Everything around it — Sqrt(Softplus) routing, top-k gate normalization,
the permute to expert-sorted rows, and the weighted combine — is plain
jnp that JAX autodiffs natively:

  - gather (rows by pair) backpropagates as a deterministic scatter-add;
  - ``segment_sum`` combine backpropagates as a gather;
  - gates get exact closed-form gradients through the normalization and
    the Sqrt(Softplus) affinity (top-k *indices* are non-diff, as usual);
  - router weights/bias gradients fall out of the same chain.

So ``jax.grad`` through this function exercises exactly the three Pallas
backward kernels (dgrad x2, swiglu-bwd) plus megablox tgmm, with no
hand-written glue derivatives to get wrong.

The EP (shard_map) variant ``moe_forward_ep_diff`` mirrors
``moe.moe_forward_ep``'s wave/bucket/dispatch logic but (a) dispatches
**bf16** token rows instead of fp8 (the a2a payload then carries clean
gradients; serving's fp8 dispatch is a bandwidth choice, an STE
optimization not needed for training correctness) and (b) runs the
diffable ``grouped_ffn`` for the expert GEMM. JAX autodiffs ``shard_map``
+ ``all_to_all`` natively — the dispatch a2a's backward is a combine-
shaped a2a and vice versa — so ``jax.grad`` through it just works.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import MoEConfig, ServingTiles
from .gemm_fp8_diff import grouped_ffn, grouped_ffn_ref


class MoEParamsTrain(NamedTuple):
    """Master (unquantized) MoE weights — what an optimizer would hold.
    Quantization to fp8 happens inside ``grouped_ffn`` each call (QAT
    semantics: master → quantize → compute, STE back to master)."""

    w_router: jax.Array        # [d, E] f32
    router_bias: jax.Array     # [E] f32 (aux-loss-free; selection only)
    w13: jax.Array             # [E, d, 2*dff]
    w2: jax.Array              # [E, dff, d]
    w13_shared: jax.Array      # [1, d, 2*dff_sh]
    w2_shared: jax.Array       # [1, dff_sh, d]


def route_train(x: jax.Array, params: MoEParamsTrain, cfg: MoEConfig):
    """Differentiable routing: returns (idx [M,k] int32, gates [M,k] f32).
    Gates carry gradients; indices do not (standard top-k routing)."""
    logits = x.astype(jnp.float32) @ params.w_router.astype(jnp.float32)
    aff = jnp.sqrt(jax.nn.softplus(logits) + 1e-20)
    sel = aff + jax.lax.stop_gradient(params.router_bias)[None, :]
    _, idx = jax.lax.top_k(jax.lax.stop_gradient(sel), cfg.topk)
    gates = jnp.take_along_axis(aff, idx, axis=-1)
    if cfg.route_norm_topk:
        gates = gates / (gates.sum(-1, keepdims=True) + 1e-20)
    return idx.astype(jnp.int32), gates


def moe_forward_local_diff(
    x: jax.Array,              # [M, d]
    params: MoEParamsTrain,
    cfg: MoEConfig,
    *,
    tiles: ServingTiles,
    idx_gates: tuple[jax.Array, jax.Array] | None = None,  # hash-routing hook
    compute_upcast: bool | None = None,
) -> jax.Array:
    """Training-path MoE forward; fully differentiable via ``jax.grad``."""
    if compute_upcast is None:
        compute_upcast = tiles.compute_upcast
    m, d = x.shape
    e, topk = cfg.n_routed, cfg.topk
    if idx_gates is None:
        idx, gates = route_train(x, params, cfg)
    else:
        idx, gates = idx_gates
    p = m * topk

    pair_expert = idx.reshape(p)
    pair_src = jnp.repeat(jnp.arange(m, dtype=jnp.int32), topk)
    pair_gate = gates.reshape(p).astype(jnp.float32)

    order = jnp.argsort(pair_expert)
    group_sizes = jnp.bincount(pair_expert, length=e).astype(jnp.int32)

    tm = tiles.gemm_tm
    m_pad = ((p + tm - 1) // tm) * tm
    src_sorted = pair_src[order]
    x_sorted = jnp.zeros((m_pad, d), x.dtype).at[:p].set(x[src_sorted])

    y = grouped_ffn(x_sorted, params.w13, params.w2, group_sizes,
                    tm=tm, compute_upcast=compute_upcast,
                    interpret=tiles.interpret)

    contrib = y[:p].astype(jnp.float32) * pair_gate[order][:, None]
    out = jax.ops.segment_sum(contrib, src_sorted, num_segments=m)

    # Shared expert: one-group FFN over all tokens.
    m_pad_tok = ((m + tm - 1) // tm) * tm
    x_tok = jnp.zeros((m_pad_tok, d), x.dtype).at[:m].set(x)
    y_sh = grouped_ffn(x_tok, params.w13_shared, params.w2_shared,
                       jnp.array([m], jnp.int32), tm=tm,
                       compute_upcast=compute_upcast,
                       interpret=tiles.interpret)
    return (out + y_sh[:m].astype(jnp.float32)).astype(x.dtype)


def moe_forward_ep_diff(
    x: jax.Array,              # [M_local, d] bf16 — this shard's tokens
    idx: jax.Array,            # [M_local, topk] int32
    gates: jax.Array,          # [M_local, topk] f32
    params: MoEParamsTrain,    # expert-sharded leaves: w13/w2 are [E_local,...]
    cfg: MoEConfig,
    *,
    axis_name: str,
    ep_size: int,
    n_waves: int,
    capacity: int,
    tiles: ServingTiles,
    compute_upcast: bool | None = None,
) -> jax.Array:
    """Training-path expert-parallel MoE — the differentiable twin of
    ``moe.moe_forward_ep``. Runs inside ``shard_map`` over ``axis_name``;
    fully autodiffable (the a2a's, argsort/scatter and segment_sum have
    native VJPs; the expert FFN is the diffable ``grouped_ffn``).

    Dispatches bf16 rows (not fp8) so gradients flow through the a2a
    cleanly; routing must be supplied (``route_train`` upstream) since
    the top-k indices are non-diff either way."""
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
    xb = x.astype(jnp.bfloat16)

    pair_expert = idx.reshape(p)
    pair_src = jnp.repeat(jnp.arange(m, dtype=jnp.int32), topk)
    pair_gate = gates.reshape(p).astype(jnp.float32)

    a2a = partial(jax.lax.all_to_all, axis_name=axis_name,
                  split_axis=0, concat_axis=0)
    out = jnp.zeros((m, d), jnp.float32)
    for w in range(n_waves):
        in_wave = (pair_expert >= w * e_wave) & (pair_expert < (w + 1) * e_wave)
        dest = jnp.where(in_wave, pair_expert // e_local, ep_size)
        sort_ix = jnp.argsort(dest)
        dest_s = dest[sort_ix]
        dcount = jnp.bincount(dest, length=ep_size + 1)
        dstart = jnp.concatenate([jnp.zeros(1, jnp.int32),
                                  jnp.cumsum(dcount)[:-1].astype(jnp.int32)])
        pos = jnp.arange(p, dtype=jnp.int32) - dstart[dest_s]
        keep = (dest_s < ep_size) & (pos < capacity)
        slot = jnp.where(keep, dest_s * capacity + pos, ep_size * capacity)
        pair_ix = sort_ix

        def scatter(payload, fill, shape_tail, dtype):
            buf = jnp.full((ep_size * capacity + 1,) + shape_tail, fill, dtype)
            return buf.at[slot].set(payload)[:-1].reshape(
                (ep_size, capacity) + shape_tail)

        # bf16 token payload (differentiable) + routing meta.
        send_x = scatter(xb[pair_src[pair_ix]], 0.0, (d,), jnp.bfloat16)
        send_e = scatter(pair_expert[pair_ix], -1, (), jnp.int32)
        send_r = scatter(pair_ix, -1, (), jnp.int32)

        recv_x, recv_e = a2a(send_x), a2a(send_e)
        rx = recv_x.reshape(ep_size * capacity, d)
        re = recv_e.reshape(ep_size * capacity)
        rel = jnp.where(re >= 0, re - my_shard * e_local, e_local)
        rorder = jnp.argsort(rel)
        m_pad = ((ep_size * capacity + tiles.gemm_tm - 1)
                 // tiles.gemm_tm) * tiles.gemm_tm
        x_sorted = jnp.zeros((m_pad, d), jnp.bfloat16).at[
            jnp.arange(ep_size * capacity)].set(rx[rorder])
        group_sizes = jnp.bincount(
            rel, length=e_local + 1)[:e_local].astype(jnp.int32)

        y = grouped_ffn(x_sorted, params.w13, params.w2, group_sizes,
                        tm=tiles.gemm_tm, compute_upcast=compute_upcast,
                        interpret=tiles.interpret)

        y_recv = jnp.zeros((ep_size * capacity, d), y.dtype).at[rorder].set(
            y[: ep_size * capacity])
        y_recv = jnp.where((re >= 0)[:, None], y_recv, 0).reshape(
            ep_size, capacity, d)

        y_back = a2a(y_recv.astype(jnp.bfloat16)).reshape(ep_size * capacity, d)
        ids_back = send_r.reshape(ep_size * capacity)
        valid = ids_back >= 0
        pair_safe = jnp.where(valid, ids_back, 0)
        contrib = (y_back.astype(jnp.float32)
                   * pair_gate[pair_safe][:, None] * valid[:, None])
        out = out + jax.ops.segment_sum(
            contrib, jnp.where(valid, pair_src[pair_safe], m),
            num_segments=m + 1)[:m]

    # Shared expert (replicated weights, no comm dependency).
    m_pad_tok = ((m + tiles.gemm_tm - 1) // tiles.gemm_tm) * tiles.gemm_tm
    x_tok = jnp.zeros((m_pad_tok, d), jnp.bfloat16).at[:m].set(xb)
    y_sh = grouped_ffn(x_tok, params.w13_shared, params.w2_shared,
                       jnp.array([m], jnp.int32), tm=tiles.gemm_tm,
                       compute_upcast=compute_upcast, interpret=tiles.interpret)
    out = out + y_sh[:m].astype(jnp.float32)
    return out.astype(x.dtype)


def moe_forward_local_diff_ref(
    x: jax.Array,
    params: MoEParamsTrain,
    cfg: MoEConfig,
    *,
    idx_gates: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """STE-modeled dense oracle; ``jax.grad`` of this grades the kernels."""
    m = x.shape[0]
    if idx_gates is None:
        idx, gates = route_train(x, params, cfg)
    else:
        idx, gates = idx_gates
    out = jnp.zeros((m, x.shape[1]), jnp.float32)
    # dense per-expert accumulation (clarity over speed)
    for g in range(cfg.n_routed):
        sel = (idx == g).astype(jnp.float32) * gates        # [M, topk]
        w_tok = sel.sum(-1, keepdims=True)                  # gate mass on g
        yg = grouped_ffn_ref(x, params.w13[g:g + 1], params.w2[g:g + 1],
                             jnp.array([m], jnp.int32))
        out = out + w_tok * yg
    out = out + grouped_ffn_ref(x, params.w13_shared, params.w2_shared,
                                jnp.array([m], jnp.int32))
    return out.astype(x.dtype)
