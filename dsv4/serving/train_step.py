"""Full-model DSv4 training step over the differentiable kernel suite.

Composes, through the whole layer schedule (SWA-intro → CSA/HCA
interleave, MoE every block, mHC-wrapped sublayers):

    Pallas custom-vjp units:  sparse_mqa_train (attention core),
                              grouped_ffn (MoE expert FFN, inside moe_diff),
                              mhc_pre_norm_diff / mhc_update_diff,
                              mhc_sinkhorn_kernel_v2 (from kernel_v2)
    Plain-jnp (native autodiff): compressors, lightning indexer, RoPE,
                              RMSNorm, routing/permute/combine, grouped
                              output projection, embedding, LM head.

Gradients therefore reach *every* trainable leaf — compressor weights via
``dK_comp`` flowing back through rms_norm→RoPE→compressor, the attention
sink via its closed-form term, mHC scale/base via the sinkhorn custom-vjp
— with two principled exceptions:

  - ``router_bias``: aux-loss-free selection bias (stop-gradient by
    construction, updated by the load-balancing controller, not SGD);
  - the lightning-indexer parameters (W_IUQ, W_w, indexer-key compressor):
    top-k *selection* is non-differentiable, and DSv4 trains the indexer
    with a separate score-distillation objective (paper §2.3.1), not the
    LM loss. ``train_step`` reports these leaves as zero-grad; the
    distillation hook is future work.

Conventions match the *serving* stack (completed-blocks-only compression,
causal SWA masking), so ``to_serving_params`` + ``model.prefill`` serve
exactly what this trains.

Memory: per-layer ``jax.checkpoint`` (on by default) keeps activation
memory at one layer's working set; inside each layer the custom-vjp units
already hold only their lean residuals (fp8 payloads, lse) — and under
remat those are recomputed, not stored.

Training KV precision: bf16 (cache quantization is a serving-time
decision; QAT covers MoE weights via grouped_ffn's STE path).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .. import eager
from .attention_train import sparse_mqa_train
from .config import ModelConfig, ServingTiles, tiles_for
from .gemm_fp8 import prepare_weight
from .mhc_diff import mhc_pre_norm_diff, mhc_update_diff
from .model import (
    LayerParams,
    MHCParams,
    ModelParams,
    _grouped_o_proj,
    _mhc_gates,
    _mixes_proj,
    _rope_at,
    layer_schedule,
)
from .moe import MoEParams, route_hash
from .moe_diff import MoEParamsTrain, moe_forward_local_diff
from .quant import quantize_weight


# ---------------------------------------------------------------------------
# Master-weight parameter tree
# ---------------------------------------------------------------------------


class TrainLayerParams(NamedTuple):
    # NOTE: no ``kind`` field — a string leaf would poison the pytree under
    # jax.value_and_grad / jax.checkpoint. The layer kind is static config,
    # derived from ``layer_schedule(cfg)`` wherever it is needed.
    attn: eager.CSAParams | eager.HCAParams
    w_o1: jax.Array
    w_o2: jax.Array
    mhc_attn: MHCParams
    mhc_moe: MHCParams
    moe: MoEParamsTrain            # master (unquantized) MoE weights


class TrainModelParams(NamedTuple):
    embed: jax.Array
    layers: tuple
    head: jax.Array


def init_train_params(key: jax.Array, cfg: ModelConfig) -> TrainModelParams:
    d, hc = cfg.d, cfg.hc
    n_mix = (2 + hc) * hc
    moe = cfg.moe

    def mk_moe(k) -> MoEParamsTrain:
        ks = jax.random.split(k, 5)
        s = d ** -0.5

        def w(kk, shape):
            return jax.random.normal(kk, shape, jnp.float32) * s

        return MoEParamsTrain(
            w_router=w(ks[0], (d, moe.n_routed)),
            router_bias=jnp.zeros((moe.n_routed,), jnp.float32),
            w13=w(ks[1], (moe.n_routed, d, 2 * moe.d_expert)),
            w2=w(ks[2], (moe.n_routed, moe.d_expert, d)),
            w13_shared=w(ks[3], (moe.n_shared, d, 2 * moe.d_expert)),
            w2_shared=w(ks[4], (moe.n_shared, moe.d_expert, d)),
        )

    def mk_mhc(k) -> MHCParams:
        return MHCParams(
            w_mix=jax.random.normal(k, (hc * d, n_mix), jnp.float32)
            * (hc * d) ** -0.5,
            scale=jnp.ones((3,), jnp.float32),
            base=jnp.zeros((n_mix,), jnp.float32),
        )

    layers = []
    for kind in layer_schedule(cfg):
        key, k0, k1, k2, k3, k4, k5 = jax.random.split(key, 7)
        if kind == "hca":
            attn = eager.init_hca_params(k0, cfg.hca)
            n_h, c = cfg.hca.n_h, cfg.hca.c
        else:
            attn = eager.init_csa_params(k0, cfg.csa)
            n_h, c = cfg.csa.n_h, cfg.csa.c
        gh = n_h // cfg.g
        layers.append(TrainLayerParams(
            attn=attn,
            w_o1=jax.random.normal(k1, (cfg.g, gh * c, cfg.d_g), jnp.float32)
            * (gh * c) ** -0.5,
            w_o2=jax.random.normal(k2, (cfg.g * cfg.d_g, d), jnp.float32)
            * (cfg.g * cfg.d_g) ** -0.5,
            mhc_attn=mk_mhc(k3), mhc_moe=mk_mhc(k4), moe=mk_moe(k5),
        ))
    key, ke, kh = jax.random.split(key, 3)
    return TrainModelParams(
        embed=jax.random.normal(ke, (cfg.vocab, d), jnp.float32) * d ** -0.5,
        layers=tuple(layers),
        head=jax.random.normal(kh, (d, cfg.vocab), jnp.float32) * d ** -0.5,
    )


def to_serving_params(tp: TrainModelParams, cfg: ModelConfig) -> ModelParams:
    """Quantize trained MoE masters into the serving format — proves the
    trained tree serves directly through ``model.prefill``."""
    def conv_moe(m: MoEParamsTrain) -> MoEParams:
        def pw(a):
            return prepare_weight(quantize_weight(a))
        return MoEParams(
            w_router=m.w_router, router_bias=m.router_bias,
            w13=pw(m.w13), w2=pw(m.w2),
            w13_shared=pw(m.w13_shared), w2_shared=pw(m.w2_shared),
        )

    layers = tuple(
        LayerParams(kind=kind, attn=lp.attn, w_o1=lp.w_o1, w_o2=lp.w_o2,
                    mhc_attn=lp.mhc_attn, mhc_moe=lp.mhc_moe,
                    moe=conv_moe(lp.moe))
        for lp, kind in zip(tp.layers, layer_schedule(cfg))
    )
    return ModelParams(embed=tp.embed, layers=layers, head=tp.head)


# ---------------------------------------------------------------------------
# Differentiable attention sublayer (prefill-shaped, serving conventions)
# ---------------------------------------------------------------------------


def _attn_train(h, lp: TrainLayerParams, kind: str, cfg: ModelConfig,
                tiles: ServingTiles):
    B, n, d = h.shape
    p = lp.attn
    acfg = cfg.hca if kind == "hca" else cfg.csa
    m = acfg.m_prime if kind == "hca" else acfg.m
    n_h, c, rd = acfg.n_h, acfg.c, acfg.rope_dim
    n_full = n // m
    positions = jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32), (B, n))
    t = jnp.arange(n)

    # SWA keys (every token; diffable preamble).
    k_swa = eager.rms_norm(_rope_at(h @ p.W_swaK, positions, rd))  # [B, n, c]

    # Compressed entries for completed blocks (diffable: dK_comp flows
    # back into the compressor weights through rope/rms).
    if kind != "swa" and n_full > 0:
        hm = h[:, : n_full * m]
        if kind == "csa":
            kc = eager.csa_compress(hm, p.W_aKV, p.W_bKV, p.W_aZ, p.W_bZ,
                                    p.B_a, p.B_b, m)
        else:
            kc = eager.hca_compress(hm, p.W_KV, p.W_Z, p.B, m)
        blk_pos = jnp.broadcast_to(jnp.arange(n_full), (B, n_full))
        kc = eager.rms_norm(_rope_at(kc, blk_pos, rd))             # [B, nf, c]
    else:
        kc = jnp.zeros((B, 1, c), h.dtype)

    # Selection (non-diff indices; serving conventions).
    if kind == "csa":
        ki = eager.csa_compress(h[:, : n_full * m], p.W_aIK, p.W_bIK,
                                p.W_aIZ, p.W_bIZ, p.B_aI, p.B_bI, m)
        scores = eager.lightning_indexer(
            h, ki, p.W_DQ, p.W_IUQ, p.W_w, m, acfg.n_I_h, acfg.c_I)
        topk = jax.lax.stop_gradient(
            eager.topk_indices(scores, acfg.topk))
    elif kind == "hca":
        all_idx = jnp.broadcast_to(
            jnp.arange(n_full, dtype=jnp.int32), (B, n, n_full))
        causal = jnp.arange(n_full)[None, :] < ((t[:, None] + 1) // m)
        topk = jnp.where(causal[None], all_idx, -1)
    else:
        topk = jnp.full((B, n, tiles.attn_chunk), -1, jnp.int32)

    q = eager.rms_norm(_rope_at(
        ((h @ p.W_DQ) @ p.W_UQ).reshape(B, n, n_h, c), positions, rd))

    o = sparse_mqa_train(q, kc, topk, k_swa, positions, p.attn_sink,
                         n_win=acfg.n_win, tiles=tiles)
    return _grouped_o_proj(o.astype(h.dtype), lp.w_o1, lp.w_o2, cfg.g)


# ---------------------------------------------------------------------------
# Full forward + loss + step
# ---------------------------------------------------------------------------


def train_forward(
    params: TrainModelParams,
    token_ids: jax.Array,      # [B, n] int32
    cfg: ModelConfig,
    *,
    tiles: ServingTiles | None = None,
    remat: bool = True,
) -> jax.Array:
    """Differentiable full-stack forward → logits ``[B, n, vocab]``."""
    if tiles is None:
        tiles = tiles_for()
    B, n = token_ids.shape
    x = params.embed[token_ids].astype(jnp.bfloat16)
    x = jnp.broadcast_to(x[:, :, None, :], (B, n, cfg.hc, cfg.d))

    kinds = layer_schedule(cfg)

    def layer_fn(x, lp: TrainLayerParams, li: int):
        # attention half (training keeps the eager mixes projection —
        # the serving-side epilogue fusion is a byte optimization only)
        pre, post, comb = _mhc_gates(_mixes_proj(x, lp.mhc_attn),
                                     lp.mhc_attn, cfg, tiles)
        h = mhc_pre_norm_diff(x, pre, tiles=tiles).astype(jnp.bfloat16)
        f = _attn_train(h, lp, kinds[li], cfg, tiles)
        x = mhc_update_diff(x, comb, post, f.astype(x.dtype), tiles=tiles)
        # MoE half
        pre, post, comb = _mhc_gates(_mixes_proj(x, lp.mhc_moe),
                                     lp.mhc_moe, cfg, tiles)
        h = mhc_pre_norm_diff(x, pre, tiles=tiles).astype(jnp.bfloat16)
        hf = h.reshape(B * n, cfg.d)
        if li < cfg.moe.n_hash_layers:
            ig = route_hash(token_ids.reshape(B * n), cfg.moe)
        else:
            ig = None
        f = moe_forward_local_diff(hf, lp.moe, cfg.moe, tiles=tiles,
                                   idx_gates=ig).reshape(B, n, cfg.d)
        return mhc_update_diff(x, comb, post, f.astype(x.dtype), tiles=tiles)

    for li, lp in enumerate(params.layers):
        fn = (jax.checkpoint(layer_fn, static_argnums=(2,)) if remat
              else layer_fn)
        x = fn(x, lp, li)

    h_out = eager.rms_norm(x.astype(jnp.float32).mean(axis=2))
    return h_out @ params.head


def lm_loss(
    params: TrainModelParams,
    token_ids: jax.Array,      # [B, n]
    cfg: ModelConfig,
    *,
    tiles: ServingTiles | None = None,
    remat: bool = True,
) -> jax.Array:
    """Next-token cross-entropy (mean over B x (n-1) positions)."""
    logits = train_forward(params, token_ids, cfg, tiles=tiles, remat=remat)
    logp = jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32), axis=-1)
    tgt = token_ids[:, 1:]
    nll = -jnp.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]
    return nll.mean()


def train_step(
    params: TrainModelParams,
    token_ids: jax.Array,
    cfg: ModelConfig,
    *,
    tiles: ServingTiles | None = None,
    remat: bool = True,
):
    """One step: returns ``(loss, grads)`` with grads matching the
    ``TrainModelParams`` pytree. Optimizer application is the caller's
    business (the paper uses Muon for matrices + AdamW for embeddings/
    norms; both consume exactly this grad tree)."""
    return jax.value_and_grad(
        lambda p: lm_loss(p, token_ids, cfg, tiles=tiles, remat=remat)
    )(params)
