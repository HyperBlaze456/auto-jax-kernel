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
    top-k *selection* is non-differentiable, so under the LM loss alone
    these leaves are zero-grad. DSv4 trains them with a separate
    score-distillation objective (paper §2.3.1): pass ``distill_weight>0``
    to ``train_step`` and the indexer learns to rank compressed blocks
    by the main attention's realized block weights (``_indexer_distill_loss``,
    a stop-gradient KL). ``distill_weight=0`` (default) keeps the
    zero-grad contract.

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

_NEG_INF = -1.0e30


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
# Indexer score-distillation (paper §2.3.1)
# ---------------------------------------------------------------------------


def _indexer_distill_loss(q, kc, scores, m, c):
    """KL( main-attention block weights ‖ indexer softmax ), the objective
    that trains the lightning indexer.

    top-k *selection* is non-differentiable, so under the LM loss the
    indexer params (W_IUQ, W_w, the ki compressor) get zero gradient. The
    indexer's job is to rank compressed blocks by how much the *real*
    attention attends them — so distill toward that: the teacher is the
    dense block-attention distribution (``softmax_s scale·q·kcᵀ``, mean
    over heads, **stop-gradient** — the indexer learns from the model, not
    vice versa), the student is ``softmax_s`` of the indexer scores over
    the same completed blocks. Gradient flows student→indexer only.

    q  [B, n, n_h, c]  attention queries (roped+normed, as the kernel sees)
    kc [B, n_full, c]  compressed attention keys (same)
    scores [B, n, n_full]  indexer scores (−inf at non-causal blocks)
    Returns a scalar mean KL over tokens with ≥1 completed block.
    """
    B, n, n_h, _ = q.shape
    n_full = kc.shape[1]
    scale = float(c) ** -0.5
    sidx = jnp.arange(n_full)
    t = jnp.arange(n)
    mask = sidx[None, :] < (t[:, None] // m)             # [n, n_full]
    neg = jnp.float32(_NEG_INF)

    tl = scale * jnp.einsum("bnhc,bsc->bnhs",
                            q.astype(jnp.float32), kc.astype(jnp.float32))
    tl = jnp.where(mask[None, :, None, :], tl, neg)
    teacher = jax.nn.softmax(tl, axis=-1).mean(axis=2)   # [B, n, n_full]
    teacher = jax.lax.stop_gradient(teacher)

    sl = jnp.where(mask[None], scores.astype(jnp.float32), neg)
    logq = jax.nn.log_softmax(sl, axis=-1)               # [B, n, n_full]
    kl = (teacher * (jnp.log(teacher + 1e-9) - logq)).sum(-1)   # [B, n]

    valid = (t // m) > 0                                 # ≥1 completed block
    kl = kl * valid[None].astype(jnp.float32)            # finite mask, no NaN
    return kl.sum() / jnp.maximum(valid.sum() * B, 1).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Differentiable attention sublayer (prefill-shaped, serving conventions)
# ---------------------------------------------------------------------------


def _attn_train(h, lp: TrainLayerParams, kind: str, cfg: ModelConfig,
                tiles: ServingTiles, distill: bool = False):
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
    scores = None
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

    # Indexer distillation: only CSA layers have a lightning indexer, and
    # only when a completed block exists to rank (paper §2.3.1).
    aux = jnp.float32(0.0)
    if distill and kind == "csa" and n_full > 0:
        aux = _indexer_distill_loss(q, kc, scores, m, acfg.c)

    o = sparse_mqa_train(q, kc, topk, k_swa, positions, p.attn_sink,
                         n_win=acfg.n_win, tiles=tiles)
    return _grouped_o_proj(o.astype(h.dtype), lp.w_o1, lp.w_o2, cfg.g), aux


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
    distill: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Differentiable full-stack forward → ``(logits [B, n, vocab],
    distill_loss)``. ``distill_loss`` is the summed indexer-distillation
    KL over CSA layers (0.0 when ``distill`` is False)."""
    if tiles is None:
        tiles = tiles_for()
    B, n = token_ids.shape
    x = params.embed[token_ids].astype(jnp.bfloat16)
    x = jnp.broadcast_to(x[:, :, None, :], (B, n, cfg.hc, cfg.d))

    kinds = layer_schedule(cfg)

    def layer_fn(carry, lp: TrainLayerParams, li: int):
        x, aux = carry
        # attention half (training keeps the eager mixes projection —
        # the serving-side epilogue fusion is a byte optimization only)
        pre, post, comb = _mhc_gates(_mixes_proj(x, lp.mhc_attn),
                                     lp.mhc_attn, cfg, tiles)
        h = mhc_pre_norm_diff(x, pre, tiles=tiles).astype(jnp.bfloat16)
        f, a = _attn_train(h, lp, kinds[li], cfg, tiles, distill=distill)
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
        x = mhc_update_diff(x, comb, post, f.astype(x.dtype), tiles=tiles)
        return (x, aux + a)

    carry = (x, jnp.float32(0.0))
    for li, lp in enumerate(params.layers):
        fn = (jax.checkpoint(layer_fn, static_argnums=(2,)) if remat
              else layer_fn)
        carry = fn(carry, lp, li)
    x, distill_loss = carry

    h_out = eager.rms_norm(x.astype(jnp.float32).mean(axis=2))
    return h_out @ params.head, distill_loss


def lm_loss(
    params: TrainModelParams,
    token_ids: jax.Array,      # [B, n]
    cfg: ModelConfig,
    *,
    tiles: ServingTiles | None = None,
    remat: bool = True,
    distill_weight: float = 0.0,
) -> jax.Array:
    """Next-token cross-entropy (mean over B x (n-1) positions), plus
    ``distill_weight`` × the indexer score-distillation KL (paper §2.3.1).

    With ``distill_weight == 0`` (default) the indexer params (W_IUQ, W_w,
    ki compressor) stay zero-grad — top-k selection is non-differentiable;
    a positive weight is what trains them, via a path independent of the
    LM cross-entropy."""
    logits, distill = train_forward(params, token_ids, cfg, tiles=tiles,
                                    remat=remat, distill=distill_weight > 0)
    logp = jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32), axis=-1)
    tgt = token_ids[:, 1:]
    nll = -jnp.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]
    return nll.mean() + distill_weight * distill


def train_step(
    params: TrainModelParams,
    token_ids: jax.Array,
    cfg: ModelConfig,
    *,
    tiles: ServingTiles | None = None,
    remat: bool = True,
    distill_weight: float = 0.0,
):
    """One step: returns ``(loss, grads)`` with grads matching the
    ``TrainModelParams`` pytree. Optimizer application is the caller's
    business (the paper uses Muon for matrices + AdamW for embeddings/
    norms; both consume exactly this grad tree).

    ``distill_weight > 0`` adds the indexer score-distillation objective,
    making the lightning-indexer leaves trainable (W_IUQ, W_w, the ki
    compressor) — otherwise they are principled zero-grad."""
    return jax.value_and_grad(
        lambda p: lm_loss(p, token_ids, cfg, tiles=tiles, remat=remat,
                          distill_weight=distill_weight)
    )(params)
