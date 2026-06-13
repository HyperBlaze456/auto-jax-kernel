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

The EP (shard_map) variant is intentionally not duplicated here: JAX
autodiffs ``shard_map`` + ``all_to_all`` natively (combine-bwd is a
dispatch-shaped a2a), so the training-EP path is ``moe.moe_forward_ep``
with ``grouped_ffn`` substituted — wiring deferred until multi-host
training is actually on the table.
"""

from __future__ import annotations

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
