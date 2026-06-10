"""Full DSv4 serving forward pass: prefill + decode over the kernel suite.

Composition per transformer block (everything mHC-wrapped, paper §2.2):

    pre, post, comb = Sinkhorn(W_mhc · X)            # kernel_v2 Pallas
    h   = mhc_pre_norm(X, pre)                       # fused Pallas
    f   = sublayer(h)                                # attention | MoE
    X   = mhc_update(X, comb, post, f)               # fused Pallas

Attention sublayer (kind ∈ {swa, csa, hca}):

    JAX/XLA:   compressors, lightning indexer + top-k, q projection,
               partial RoPE, RMSNorm, KV quantization (cache write)
    Pallas:    sparse_mqa_gathered (fused gather + flash + sink)
    JAX/XLA:   grouped output projection (g groups → d_g → d)

MoE sublayer: route (fp32) → moe_forward_local / moe_forward_ep
(Pallas grouped FP8 GEMMs + fused SwiGLU-quant inside).

Serving conventions (deliberate, documented deviations from `eager`)
--------------------------------------------------------------------

1. **Completed-blocks-only compression.** A compressed entry exists once
   its source block is complete; queries attend entries with block index
   ``< t // m`` (CSA — identical to the eager indexer mask) and
   ``< (t+1) // m'`` (HCA — eager's full-sequence compressor instead
   builds entries containing intra-block *future* tokens, which a decoder
   cannot do). The SWA branch covers the in-progress window either way.
2. **Causally-masked SWA edge** (no zero-pad pseudo-keys) — see
   attention.py.
3. **KV cache stores post-RoPE, post-RMSNorm entries** in the hybrid
   fp8/bf16 format, so decode never re-touches old tokens.
4. **mHC readout**: final hidden = stream-mean of X (the paper does not
   specify the serving readout; swap for a learned C_final if the
   checkpoint provides one). The mixes projection input is the
   RMSNorm'd flattened stream (hc·d → (2+hc)·hc = 24 outputs — the
   paper's "output dimension of only 24" GEMM).

Decode cache mechanics
----------------------

Per layer the cache holds: quantized compressed entries (+ a count), the
quantized raw SWA ring, indexer keys (CSA), and the last ``2m`` (CSA) /
``m'`` (HCA) raw hidden rows. Every step appends one SWA row; when a
block boundary is crossed, the new entry is computed *from the hidden
ring only* (one tiny 2m·d GEMM set), RoPE'd at its block position,
normalized, quantized, appended. Existing entries are never touched —
decode HBM traffic per layer is exactly: gather reads + one SWA row write
+ (amortized 1/m) one entry write.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .. import eager
from ..kernel_config import KernelConfig
from ..kernel_v2 import mhc_sinkhorn_kernel_v2
from .attention import sparse_mqa_gathered
from .config import ModelConfig, ServingTiles, tiles_for
from .gemm_fp8 import prepare_weight
from .mhc import mhc_pre_norm, mhc_update
from .moe import MoEParams, moe_forward_local, route, route_hash
from .quant import KVQuant, quantize_kv, quantize_weight


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


class MHCParams(NamedTuple):
    w_mix: jax.Array       # [hc*d, (2+hc)*hc]
    scale: jax.Array       # [3]
    base: jax.Array        # [(2+hc)*hc]


class LayerParams(NamedTuple):
    kind: str                      # "swa" | "csa" | "hca"  (static)
    attn: eager.CSAParams | eager.HCAParams
    w_o1: jax.Array                # [g, (n_h//g)*c, d_g]
    w_o2: jax.Array                # [g*d_g, d]
    mhc_attn: MHCParams
    mhc_moe: MHCParams
    moe: MoEParams


class ModelParams(NamedTuple):
    embed: jax.Array               # [vocab, d]
    layers: tuple                  # tuple[LayerParams]
    head: jax.Array                # [d, vocab]


def layer_schedule(cfg: ModelConfig) -> list[str]:
    sched = []
    for li in range(cfg.n_layers):
        if li < cfg.n_swa_only:
            sched.append(cfg.intro_kind)
        else:
            sched.append("csa" if (li - cfg.n_swa_only) % 2 == 0 else "hca")
    return sched


def init_params(key: jax.Array, cfg: ModelConfig) -> ModelParams:
    """Random init in the serving formats (expert weights pre-quantized)."""
    d, hc = cfg.d, cfg.hc
    n_mix = (2 + hc) * hc
    moe = cfg.moe

    def mk_moe(k) -> MoEParams:
        ks = jax.random.split(k, 6)
        s = d ** -0.5

        def w(kk, shape):
            return jax.random.normal(kk, shape, jnp.float32) * s

        def pw(a):
            return prepare_weight(quantize_weight(a))
        return MoEParams(
            w_router=w(ks[0], (d, moe.n_routed)),
            router_bias=jnp.zeros((moe.n_routed,), jnp.float32),
            w13=pw(w(ks[1], (moe.n_routed, d, 2 * moe.d_expert))),
            w2=pw(w(ks[2], (moe.n_routed, moe.d_expert, d))),
            w13_shared=pw(w(ks[3], (moe.n_shared, d, 2 * moe.d_expert))),
            w2_shared=pw(w(ks[4], (moe.n_shared, moe.d_expert, d))),
        )

    def mk_mhc(k) -> MHCParams:
        return MHCParams(
            w_mix=jax.random.normal(k, (hc * d, n_mix), jnp.float32) * (hc * d) ** -0.5,
            scale=jnp.ones((3,), jnp.float32),
            base=jnp.zeros((n_mix,), jnp.float32),
        )

    layers = []
    for _li, kind in enumerate(layer_schedule(cfg)):
        key, k0, k1, k2, k3, k4, k5 = jax.random.split(key, 7)
        if kind == "hca":
            attn = eager.init_hca_params(k0, cfg.hca)
            n_h, c = cfg.hca.n_h, cfg.hca.c
        else:
            attn = eager.init_csa_params(k0, cfg.csa)
            n_h, c = cfg.csa.n_h, cfg.csa.c
        gh = n_h // cfg.g
        layers.append(LayerParams(
            kind=kind,
            attn=attn,
            w_o1=jax.random.normal(k1, (cfg.g, gh * c, cfg.d_g), jnp.float32)
            * (gh * c) ** -0.5,
            w_o2=jax.random.normal(k2, (cfg.g * cfg.d_g, d), jnp.float32)
            * (cfg.g * cfg.d_g) ** -0.5,
            mhc_attn=mk_mhc(k3),
            mhc_moe=mk_mhc(k4),
            moe=mk_moe(k5),
        ))
    key, ke, kh = jax.random.split(key, 3)
    return ModelParams(
        embed=jax.random.normal(ke, (cfg.vocab, d), jnp.float32) * d ** -0.5,
        layers=tuple(layers),
        head=jax.random.normal(kh, (d, cfg.vocab), jnp.float32) * d ** -0.5,
    )


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------


class LayerCache(NamedTuple):
    kc: KVQuant            # compressed entries  [B, S_blk, ...]
    swa: KVQuant           # raw SWA keys        [B, S_max, ...]
    ki: jax.Array          # indexer keys        [B, S_blk, c_I] (csa) or [B,1,1]
    h_ring: jax.Array      # last `ring` hidden rows [B, ring, d]


class ServingState(NamedTuple):
    pos: jax.Array         # [] int32 — tokens generated so far
    caches: tuple          # tuple[LayerCache]


def _empty_kv(B, S, c, rope_dim):
    return KVQuant(
        nope=jnp.zeros((B, S, c - rope_dim), jnp.float8_e4m3fn),
        rope=jnp.zeros((B, S, rope_dim), jnp.bfloat16),
        scale=jnp.zeros((B, S, 1), jnp.float32),
    )


def init_state(cfg: ModelConfig, B: int, s_max: int) -> ServingState:
    caches = []
    for kind in layer_schedule(cfg):
        if kind == "hca":
            m, c, rd = cfg.hca.m_prime, cfg.hca.c, cfg.hca.rope_dim
            ki = jnp.zeros((B, 1, 1), jnp.bfloat16)
            ring = m
        else:
            m, c, rd = cfg.csa.m, cfg.csa.c, cfg.csa.rope_dim
            ki = jnp.zeros((B, s_max // m, cfg.csa.c_I), jnp.bfloat16)
            ring = 2 * m
        caches.append(LayerCache(
            kc=_empty_kv(B, s_max // m, c, rd),
            swa=_empty_kv(B, s_max, c, rd),
            ki=ki,
            h_ring=jnp.zeros((B, ring, cfg.d), jnp.bfloat16),
        ))
    return ServingState(pos=jnp.zeros((), jnp.int32), caches=tuple(caches))


def _kv_set(kv: KVQuant, row: KVQuant, at) -> KVQuant:
    """Write `row` ([B, w, ...]) into the caches at position `at`."""
    def upd(a, v):
        return jax.lax.dynamic_update_slice(a, v.astype(a.dtype), (0, at, 0))
    return KVQuant(nope=upd(kv.nope, row.nope), rope=upd(kv.rope, row.rope),
                   scale=upd(kv.scale, row.scale))


# ---------------------------------------------------------------------------
# Attention sublayer pieces (XLA preamble shared by prefill & decode)
# ---------------------------------------------------------------------------


def _rope_at(x: jax.Array, positions: jax.Array, rope_dim: int) -> jax.Array:
    """Partial RoPE at explicit integer positions.

    ``x``: ``[..., L, hd]`` or ``[..., L, H, hd]``; ``positions``:
    ``[..., L]``. Matches ``eager.apply_partial_rope`` for contiguous
    positions but takes arbitrary absolute positions (decode)."""
    half = rope_dim // 2
    inv = 1.0 / (10000.0 ** (jnp.arange(half, dtype=jnp.float32) * 2.0 / rope_dim))
    ang = positions.astype(jnp.float32)[..., None] * inv      # [..., L, half]
    cos, sin = jnp.cos(ang), jnp.sin(ang)
    no = x[..., : x.shape[-1] - rope_dim]
    ro = x[..., x.shape[-1] - rope_dim:]
    x1, x2 = ro[..., :half], ro[..., half:]
    for _ in range(x1.ndim - cos.ndim):       # add head axes before `half`
        cos, sin = cos[..., None, :], sin[..., None, :]
    rot = jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
    return jnp.concatenate([no, rot], -1)


def _q_proj(h, p, n_h, c, rope_dim, positions):
    B, T, _ = h.shape
    q = ((h @ p.W_DQ) @ p.W_UQ).reshape(B, T, n_h, c)
    q = _rope_at(q, positions, rope_dim)
    return eager.rms_norm(q)


def _swa_rows(h, p, rope_dim, positions):
    """Per-token SWA key rows (roped + normed), ready for cache write."""
    k = h @ p.W_swaK
    k = _rope_at(k, positions, rope_dim)
    return eager.rms_norm(k)


def _grouped_o_proj(o, w_o1, w_o2, g):
    B, T, n_h, c = o.shape
    og = o.reshape(B, T, g, (n_h // g) * c)
    inter = jnp.einsum("btgk,gkj->btgj", og, w_o1.astype(o.dtype))
    return inter.reshape(B, T, -1) @ w_o2.astype(o.dtype)


def _csa_entry_from_blocks(h2m, p, m, rope_dim, blk_pos):
    """One CSA compressed entry from its 2m source rows ([B, 2m, d]):
    blocks (i-1, i) with the overlapped softmax mix, then RoPE at the
    block position and RMSNorm. ``blk_pos < 0`` is treated as block 0
    (the i=0 boundary: the b-path is -inf-masked)."""
    Ca = h2m @ p.W_aKV
    Cb = h2m @ p.W_bKV
    Za = h2m @ p.W_aZ
    Zb = h2m @ p.W_bZ
    first = blk_pos == 0
    za = Za[:, m:] + p.B_a[None]                          # current block
    zb = jnp.where(first, -jnp.inf, Zb[:, :m] + p.B_b[None])
    logits = jnp.concatenate([za, zb], axis=1)            # [B, 2m, c]
    w = jax.nn.softmax(logits, axis=1)
    cb_prev = jnp.where(first, 0.0, Cb[:, :m])
    entry = (w[:, :m] * Ca[:, m:]).sum(1) + (w[:, m:] * cb_prev).sum(1)
    entry = _rope_at(entry[:, None, :], jnp.full((1, 1), blk_pos), rope_dim)[:, 0]
    return eager.rms_norm(entry)                          # [B, c]


def _csa_ientry_from_blocks(h2m, p, m, blk_pos):
    """Indexer-key compressed entry (same compressor, I-weights, no rope)."""
    Ca = h2m @ p.W_aIK
    Cb = h2m @ p.W_bIK
    Za = h2m @ p.W_aIZ
    Zb = h2m @ p.W_bIZ
    first = blk_pos == 0
    za = Za[:, m:] + p.B_aI[None]
    zb = jnp.where(first, -jnp.inf, Zb[:, :m] + p.B_bI[None])
    w = jax.nn.softmax(jnp.concatenate([za, zb], axis=1), axis=1)
    cb_prev = jnp.where(first, 0.0, Cb[:, :m])
    return (w[:, :m] * Ca[:, m:]).sum(1) + (w[:, m:] * cb_prev).sum(1)


def _hca_entry_from_block(hm, p, rope_dim, blk_pos):
    """One HCA compressed entry from its m' source rows ([B, m', d])."""
    C = hm @ p.W_KV
    Z = hm @ p.W_Z
    w = jax.nn.softmax(Z + p.B[None], axis=1)
    entry = (w * C).sum(1)
    entry = _rope_at(entry[:, None, :], jnp.full((1, 1), blk_pos), rope_dim)[:, 0]
    return eager.rms_norm(entry)


def _indexer_scores_step(h_t, ki_cache, p, cfg_csa, n_valid):
    """Lightning indexer for one decode token: [B, 1, S_blk] scores."""
    B = h_t.shape[0]
    cQ = h_t @ p.W_DQ                                     # [B, 1, d_c]
    qI = (cQ @ p.W_IUQ).reshape(B, 1, cfg_csa.n_I_h, cfg_csa.c_I)
    wI = h_t @ p.W_w                                      # [B, 1, n_I_h]
    qk = jnp.einsum("bthc,bsc->btsh", qI.astype(jnp.float32),
                    ki_cache.astype(jnp.float32))
    scores = (wI.astype(jnp.float32)[:, :, None, :] * jax.nn.relu(qk)).sum(-1)
    s = jnp.arange(ki_cache.shape[1])
    return jnp.where((s < n_valid)[None, None, :], scores, -jnp.inf)


# ---------------------------------------------------------------------------
# Sublayer drivers
# ---------------------------------------------------------------------------


def _attn_prefill(h, lp: LayerParams, cfg: ModelConfig, cache: LayerCache,
                  tiles: ServingTiles):
    """Prefill attention for one layer. Returns (f_out, updated cache)."""
    B, n, d = h.shape
    kind = lp.kind
    p = lp.attn
    positions = jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32), (B, n))

    if kind == "hca":
        acfg = cfg.hca
        m = acfg.m_prime
    else:
        acfg = cfg.csa
        m = acfg.m
    n_h, c, rd = acfg.n_h, acfg.c, acfg.rope_dim
    n_full = n // m                                       # completed blocks

    # --- SWA rows → cache ---
    swa_rows = _swa_rows(h, p, rd, positions)             # [B, n, c]
    cache = cache._replace(swa=_kv_set(cache.swa, quantize_kv(swa_rows, rd), 0))

    # --- compressed entries (completed blocks only) → cache ---
    if kind != "swa" and n_full > 0:
        hm = h[:, : n_full * m]
        if kind == "csa":
            kc = eager.csa_compress(hm, p.W_aKV, p.W_bKV, p.W_aZ, p.W_bZ,
                                    p.B_a, p.B_b, m)
            ki = eager.csa_compress(hm, p.W_aIK, p.W_bIK, p.W_aIZ, p.W_bIZ,
                                    p.B_aI, p.B_bI, m)
            cache = cache._replace(
                ki=jax.lax.dynamic_update_slice(
                    cache.ki, ki.astype(cache.ki.dtype), (0, 0, 0)))
        else:
            kc = eager.hca_compress(hm, p.W_KV, p.W_Z, p.B, m)
        blk_pos = jnp.broadcast_to(jnp.arange(n_full), (B, n_full))
        kc = _rope_at(kc, blk_pos, rd)
        kc = eager.rms_norm(kc)
        cache = cache._replace(kc=_kv_set(cache.kc, quantize_kv(kc, rd), 0))

    # --- ring of trailing raw hidden rows (for decode continuation) ---
    ring = cache.h_ring.shape[1]
    tail = h[:, -ring:] if n >= ring else jnp.pad(h, ((0, 0), (ring - n, 0), (0, 0)))
    cache = cache._replace(h_ring=tail.astype(cache.h_ring.dtype))

    # --- per-token selection set ---
    t = jnp.arange(n)
    if kind == "csa":
        scores = eager.lightning_indexer(
            h, cache.ki[:, :max(n_full, 1)].astype(h.dtype),
            p.W_DQ, p.W_IUQ, p.W_w, m, acfg.n_I_h, acfg.c_I)
        topk = eager.topk_indices(scores, acfg.topk)      # [B, n, k]
    elif kind == "hca":
        s_blk = cache.kc.nope.shape[1]
        all_idx = jnp.broadcast_to(jnp.arange(s_blk, dtype=jnp.int32),
                                   (B, n, s_blk))
        causal = jnp.arange(s_blk)[None, :] < ((t[:, None] + 1) // m)
        topk = jnp.where(causal[None], all_idx, -1)
    else:  # swa-only
        topk = jnp.full((B, n, tiles.attn_chunk), -1, jnp.int32)

    q = _q_proj(h, p, n_h, c, rd, positions)
    o = sparse_mqa_gathered(q, cache.kc, topk, cache.swa, positions,
                            p.attn_sink, n_win=acfg.n_win, rope_dim=rd,
                            tiles=tiles)
    return _grouped_o_proj(o, lp.w_o1, lp.w_o2, cfg.g), cache


def _attn_decode(h_t, pos, lp: LayerParams, cfg: ModelConfig,
                 cache: LayerCache, tiles: ServingTiles):
    """One-token attention step. ``pos`` is the absolute position of the
    incoming token (cache already holds rows for 0..pos-1)."""
    B = h_t.shape[0]
    kind = lp.kind
    p = lp.attn
    if kind == "hca":
        acfg, m = cfg.hca, cfg.hca.m_prime
    else:
        acfg, m = cfg.csa, cfg.csa.m
    n_h, c, rd = acfg.n_h, acfg.c, acfg.rope_dim
    positions = jnp.broadcast_to(pos[None, None], (B, 1)).astype(jnp.int32)

    # --- append SWA row ---
    swa_row = _swa_rows(h_t, p, rd, positions)            # [B, 1, c]
    cache = cache._replace(swa=_kv_set(cache.swa, quantize_kv(swa_row, rd), pos))

    # --- roll hidden ring, maybe emit a compressed entry ---
    h_ring = jnp.concatenate(
        [cache.h_ring[:, 1:], h_t.astype(cache.h_ring.dtype)], axis=1)
    cache = cache._replace(h_ring=h_ring)

    new_block = (pos + 1) % m == 0                        # block just completed
    blk_id = (pos + 1) // m - 1                           # its index
    if kind != "swa":
        hr = h_ring.astype(h_t.dtype)
        if kind == "csa":
            entry = _csa_entry_from_blocks(hr, p, m, rd, blk_id)      # [B, c]
            ient = _csa_ientry_from_blocks(hr, p, m, blk_id)          # [B, c_I]
        else:
            entry = _hca_entry_from_block(hr[:, -m:], p, rd, blk_id)
            ient = None
        # write at blk_id only when a block completed; else rewrite row 0
        # with its own current value (no-op).
        at = jnp.where(new_block, blk_id, 0)
        row = quantize_kv(entry[:, None, :], rd)
        keep = jax.tree.map(
            lambda a: jax.lax.dynamic_slice(a, (0, at, 0), (B, 1, a.shape[-1])),
            cache.kc)
        def sel(nv, old):
            return jax.tree.map(
                lambda x, y: jnp.where(new_block, x, y.astype(x.dtype)), nv, old)
        cache = cache._replace(kc=_kv_set(cache.kc, sel(row, keep), at))
        if ient is not None:
            old = jax.lax.dynamic_slice(
                cache.ki, (0, at, 0), (B, 1, cache.ki.shape[-1]))
            newv = jnp.where(new_block, ient[:, None, :].astype(cache.ki.dtype), old)
            cache = cache._replace(
                ki=jax.lax.dynamic_update_slice(cache.ki, newv, (0, at, 0)))

    n_blk_valid = (pos + 1) // m                          # completed blocks now

    # --- selection ---
    if kind == "csa":
        n_attend = pos // m                               # eager indexer mask
        scores = _indexer_scores_step(h_t, cache.ki, p, acfg, n_attend)
        topk = eager.topk_indices(scores, acfg.topk)
    elif kind == "hca":
        s_blk = cache.kc.nope.shape[1]
        idx = jnp.arange(s_blk, dtype=jnp.int32)
        topk = jnp.where(idx < n_blk_valid, idx, -1)[None, None, :]
        topk = jnp.broadcast_to(topk, (B, 1, s_blk))
    else:
        topk = jnp.full((B, 1, tiles.attn_chunk), -1, jnp.int32)

    q = _q_proj(h_t, p, n_h, c, rd, positions)
    o = sparse_mqa_gathered(q, cache.kc, topk, cache.swa, positions,
                            p.attn_sink, n_win=acfg.n_win, rope_dim=rd,
                            tiles=tiles)
    return _grouped_o_proj(o, lp.w_o1, lp.w_o2, cfg.g), cache


def _moe_sublayer(h, token_ids, layer_idx, lp: LayerParams, cfg: ModelConfig,
                  tiles: ServingTiles):
    B, T, d = h.shape
    x = h.reshape(B * T, d)
    if layer_idx < cfg.moe.n_hash_layers:
        idx, gates = route_hash(token_ids.reshape(B * T), cfg.moe)
    else:
        idx, gates = route(x, lp.moe, cfg.moe)
    return moe_forward_local(x, idx, gates, lp.moe, cfg.moe,
                             tiles=tiles).reshape(B, T, d)


def _mhc_gates(x, mp: MHCParams, cfg: ModelConfig, tiles: ServingTiles):
    """mixes projection + Sinkhorn → (pre, post, comb). fp32 throughout."""
    B, n, hc, d = x.shape
    flat = eager.rms_norm(x.astype(jnp.float32)).reshape(B, n, hc * d)
    mixes = flat @ mp.w_mix
    kc = KernelConfig(bq=1, bs=128, bn_sinkhorn=tiles.mhc_bn,
                      interpret=tiles.interpret)
    return mhc_sinkhorn_kernel_v2(mixes, mp.scale, mp.base, hc,
                                  cfg.sinkhorn_iters, 1e-6, config=kc)


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def _block(x, h_fn, mp: MHCParams, cfg, tiles):
    """One mHC-wrapped sublayer: X ← comb·X + post⊗F(prenorm(pre·X))."""
    pre, post, comb = _mhc_gates(x, mp, cfg, tiles)
    h = mhc_pre_norm(x, pre, tiles=tiles)
    f = h_fn(h.astype(jnp.bfloat16))
    return mhc_update(x, comb, post, f.astype(x.dtype), tiles=tiles)


def prefill(
    params: ModelParams,
    token_ids: jax.Array,      # [B, n] int32
    cfg: ModelConfig,
    state: ServingState,
    *,
    tiles: ServingTiles | None = None,
):
    """Full prefill. Returns ``(logits [B, n, vocab], state)``."""
    if tiles is None:
        tiles = tiles_for()
    B, n = token_ids.shape
    x = params.embed[token_ids].astype(jnp.bfloat16)       # [B, n, d]
    x = jnp.broadcast_to(x[:, :, None, :], (B, n, cfg.hc, cfg.d))

    caches = list(state.caches)
    for li, lp in enumerate(params.layers):
        def attn_fn(h, _li=li, _lp=lp):
            f, caches[_li] = _attn_prefill(h, _lp, cfg, caches[_li], tiles)
            return f
        x = _block(x, attn_fn, lp.mhc_attn, cfg, tiles)
        x = _block(x, lambda h, _li=li, _lp=lp: _moe_sublayer(
            h, token_ids, _li, _lp, cfg, tiles), lp.mhc_moe, cfg, tiles)

    h_out = eager.rms_norm(x.astype(jnp.float32).mean(axis=2))
    logits = h_out @ params.head
    return logits, ServingState(pos=jnp.int32(n), caches=tuple(caches))


def decode_step(
    params: ModelParams,
    token_id: jax.Array,       # [B] int32 — the incoming token
    cfg: ModelConfig,
    state: ServingState,
    *,
    tiles: ServingTiles | None = None,
):
    """One decode step. Returns ``(logits [B, vocab], state)``."""
    if tiles is None:
        tiles = tiles_for()
    B = token_id.shape[0]
    pos = state.pos
    x = params.embed[token_id].astype(jnp.bfloat16)[:, None, :]
    x = jnp.broadcast_to(x[:, :, None, :], (B, 1, cfg.hc, cfg.d))

    caches = list(state.caches)
    for li, lp in enumerate(params.layers):
        def attn_fn(h, _li=li, _lp=lp):
            f, caches[_li] = _attn_decode(h, pos, _lp, cfg, caches[_li], tiles)
            return f
        x = _block(x, attn_fn, lp.mhc_attn, cfg, tiles)
        x = _block(x, lambda h, _li=li, _lp=lp: _moe_sublayer(
            h, token_id[:, None], _li, _lp, cfg, tiles), lp.mhc_moe, cfg, tiles)

    h_out = eager.rms_norm(x.astype(jnp.float32).mean(axis=2))
    logits = (h_out @ params.head)[:, 0]
    return logits, ServingState(pos=pos + 1, caches=tuple(caches))
