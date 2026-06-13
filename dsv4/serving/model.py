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
raw SWA window as a **dual-write unrolled ring** of ``2*n_win`` rows
(position p at slots ``p % n_win`` and ``p % n_win + n_win``, so the
window is always one contiguous position-ordered slab — HARDWARE_NOTES
§13.2; the old full-``s_max`` allocation was 25 GB/seq at 1M Flash),
fp8 indexer keys (CSA, §13.1), per-page indexer summaries (CSA with
``csa_pages > 0``, §13.3), and the last ``2m`` (CSA) / ``m'`` (HCA) raw
hidden rows. Every step appends one SWA row (two slot writes); when a
block boundary is crossed, the new entry is computed *from the hidden
ring only* (one tiny 2m·d GEMM set), RoPE'd at its block position,
normalized, quantized, appended; when a *page* of ``csa_pages`` entries
completes, its summary row is recomputed from the stored fp8 rows.
Existing entries are never touched — decode HBM traffic per layer is
exactly: gather reads + indexer scan + two SWA row writes + (amortized
1/m) one entry write (+ 1/(m·P) one summary write).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .. import eager
from ..kernel_config import KernelConfig
from ..kernel_v2 import mhc_sinkhorn_kernel_v2
from .attention import sparse_mqa_gathered
from .attention_paged import sparse_mqa_paged
from .indexer_scan import indexer_scores_decode, indexer_ub_scores_decode
from .config import ModelConfig, ServingTiles, tiles_for
from .gemm_fp8 import prepare_weight
from .mhc import mhc_pre_norm, mhc_update, mhc_update_mix
from .moe import MoEParams, moe_forward_local, route, route_hash
from .quant import (KVQuant, RowQuant, dequantize_rows, quantize_kv,
                    quantize_rows, quantize_rows_bound, quantize_weight)


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
    swa: KVQuant           # raw SWA ring        [B, 2*n_win, ...]
    ki: RowQuant           # indexer keys, fp8/row [B, S_blk, c_I] (csa) or [B,1,1]
    kis: RowQuant          # page summaries [B, S_blk/P, c_I] (csa, pages) or [B,1,1]
    #                        — mean rows, or the MAX envelope in exact mode
    kis_lo: RowQuant       # page MIN envelope (csa, pages, exact) or [B,1,1]
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


def _empty_rows(B, S, c):
    return RowQuant(q=jnp.zeros((B, S, c), jnp.float8_e4m3fn),
                    scale=jnp.zeros((B, S, 1), jnp.float32))


def init_state(cfg: ModelConfig, B: int, s_max: int) -> ServingState:
    if cfg.csa_pages_exact and cfg.csa_pages == 0:
        raise ValueError("csa_pages_exact requires csa_pages > 0")
    caches = []
    for kind in layer_schedule(cfg):
        P = cfg.csa_pages
        if kind == "hca":
            m, c, rd = cfg.hca.m_prime, cfg.hca.c, cfg.hca.rope_dim
            n_win = cfg.hca.n_win
            ki = _empty_rows(B, 1, 1)
            kis = kis_lo = _empty_rows(B, 1, 1)
            ring = m
        else:
            m, c, rd = cfg.csa.m, cfg.csa.c, cfg.csa.rope_dim
            n_win = cfg.csa.n_win
            ki = _empty_rows(B, s_max // m, cfg.csa.c_I)
            kis = kis_lo = _empty_rows(B, 1, 1)
            if P and kind == "csa":
                if cfg.csa.topk % P or (s_max // m) % P:
                    raise ValueError(
                        f"csa_pages={P} must divide topk={cfg.csa.topk} and "
                        f"the entry capacity {s_max // m}")
                if P * m > cfg.csa.n_win:
                    raise ValueError(
                        f"P*m = {P * m} must be <= n_win={cfg.csa.n_win} so "
                        "the in-progress page stays inside the SWA window")
                kis = _empty_rows(B, s_max // m // P, cfg.csa.c_I)
                if cfg.csa_pages_exact:
                    R = cfg.csa_rescan or 2 * cfg.csa.topk // P
                    if R * P < cfg.csa.topk:
                        raise ValueError(
                            f"csa_rescan={R} pages cover {R * P} rows < "
                            f"topk={cfg.csa.topk}")
                    if R > s_max // m // P:
                        raise ValueError(
                            f"csa_rescan={R} exceeds the page capacity "
                            f"{s_max // m // P}")
                    kis_lo = _empty_rows(B, s_max // m // P, cfg.csa.c_I)
            ring = 2 * m
        caches.append(LayerCache(
            kc=_empty_kv(B, s_max // m, c, rd),
            # Raw SWA rows live in a dual-write unrolled ring: 2*n_win rows
            # instead of s_max (only the trailing n_win window is ever
            # read — the full-s_max allocation was the 1M capacity blocker,
            # HARDWARE_NOTES §11.1: 580 B x S x n_layers ~ 25 GB/seq).
            swa=_empty_kv(B, 2 * n_win, c, rd),
            ki=ki,
            kis=kis,
            kis_lo=kis_lo,
            h_ring=jnp.zeros((B, ring, cfg.d), jnp.bfloat16),
        ))
    return ServingState(pos=jnp.zeros((), jnp.int32), caches=tuple(caches))


def _kv_set(kv: KVQuant, row: KVQuant, at) -> KVQuant:
    """Write `row` ([B, w, ...]) into the caches at position `at`."""
    def upd(a, v):
        return jax.lax.dynamic_update_slice(a, v.astype(a.dtype), (0, at, 0))
    return KVQuant(nope=upd(kv.nope, row.nope), rope=upd(kv.rope, row.rope),
                   scale=upd(kv.scale, row.scale))


def _row_set(rq: RowQuant, row: RowQuant, at) -> RowQuant:
    """Write quantized rows ([B, w, ...]) into a RowQuant cache at `at`."""
    def upd(a, v):
        return jax.lax.dynamic_update_slice(a, v.astype(a.dtype), (0, at, 0))
    return RowQuant(q=upd(rq.q, row.q), scale=upd(rq.scale, row.scale))


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
    """Lightning indexer for one decode token: [B, 1, S_blk] scores.

    Eager reference / test oracle. The decode hot path uses
    ``indexer_scan.indexer_scores_decode`` — same math, but the Pallas
    kernel reads only the ``n_valid``-row prefix from HBM, where this
    einsum's static shapes scan the whole allocation.

    ``ki_cache`` is a dequantized fp32 array [B, S_blk, c_I] — callers
    dequantize their RowQuant cache on the way in (element-wise, fuses
    into the scan; HBM bytes stay fp8)."""
    B = h_t.shape[0]
    cQ = h_t @ p.W_DQ                                     # [B, 1, d_c]
    qI = (cQ @ p.W_IUQ).reshape(B, 1, cfg_csa.n_I_h, cfg_csa.c_I)
    wI = h_t @ p.W_w                                      # [B, 1, n_I_h]
    qk = jnp.einsum("bthc,bsc->btsh", qI.astype(jnp.float32),
                    ki_cache.astype(jnp.float32))
    scores = (wI.astype(jnp.float32)[:, :, None, :] * jax.nn.relu(qk)).sum(-1)
    s = jnp.arange(ki_cache.shape[1])
    return jnp.where((s < n_valid)[None, None, :], scores, -jnp.inf)


def _rescan_rows(h_t, ki: RowQuant, p, cfg_csa, pages, P, n_valid):
    """Exact f32 indexer scores for the rows of the selected pages (§13.7).

    ``pages`` [B, 1, R] int32 page ids (-1 invalid). Gathers the R*P fp8
    rows from the ki cache and scores them with the same math as
    ``_indexer_scores_step`` — the candidates' scores are therefore
    *identical* to what a full row scan would have produced. Returns
    ``(scores [B, 1, R*P], rows [B, R*P])`` with -inf at invalid rows.
    """
    B = h_t.shape[0]
    R = pages.shape[-1]
    S_blk = ki.q.shape[1]
    raw = pages[:, 0][..., None] * P + jnp.arange(P)      # [B, R, P]
    valid = jnp.logical_and(pages[:, 0][..., None] >= 0,
                            raw < n_valid).reshape(B, R * P)
    rows = jnp.clip(raw, 0, S_blk - 1).reshape(B, R * P).astype(jnp.int32)
    kf = dequantize_rows(RowQuant(
        q=jnp.take_along_axis(ki.q, rows[..., None], axis=1),
        scale=jnp.take_along_axis(ki.scale, rows[..., None], axis=1)))
    cQ = h_t @ p.W_DQ
    qI = (cQ @ p.W_IUQ).reshape(B, 1, cfg_csa.n_I_h, cfg_csa.c_I)
    wI = h_t @ p.W_w
    qk = jnp.einsum("bthc,bsc->btsh", qI.astype(jnp.float32), kf)
    sc = (wI.astype(jnp.float32)[:, :, None, :] * jax.nn.relu(qk)).sum(-1)
    return jnp.where(valid[:, None, :], sc, -jnp.inf), rows


def _topk_rows(scores, rows, k):
    """Top-k over rescan candidates, mapped back to global row ids.

    ``scores`` [B, 1, R*P], ``rows`` [B, R*P] → [B, 1, k] int32 with -1
    for invalid (-inf) picks — the ``topk_indices`` sentinel contract.
    """
    _, loc = jax.lax.top_k(scores, k)                     # [B, 1, k]
    sel = jnp.take_along_axis(scores, loc, axis=-1)
    glob = jnp.take_along_axis(rows[:, None, :], loc, axis=-1)
    return jnp.where(jnp.isfinite(sel), glob.astype(jnp.int32), -1)


def _exact_certificate(ub, pages, scores, k):
    """[B, 1] bool: the §13.7 rescan candidate set provably covers the
    row-exact top-k — the k-th best candidate score dominates every
    unselected page's upper bound (so no outside row can belong).

    Not on the hot path (selection is already made); serving can sample
    it to size ``csa_rescan``, tests assert the implication with it.
    """
    S_pg = ub.shape[-1]
    sel = (jnp.arange(S_pg)[None, None, :, None]
           == pages[..., None, :]).any(-1)               # [B, 1, S_pg]
    rest_ub = jnp.where(sel, -jnp.inf, ub).max(-1)       # [B, 1]
    kth = jax.lax.top_k(scores, k)[0][..., -1]           # [B, 1]
    return kth >= rest_ub


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

    # --- SWA rows: full-length transient (this prefill attends them all);
    # only the trailing-window ring persists in the cache ---
    n_win = acfg.n_win
    swa_q = quantize_kv(_swa_rows(h, p, rd, positions), rd)   # [B, n, ...]

    def _ring(a):
        # Position-order tail (front zero-pad if n < n_win), rolled so the
        # row of position p lands at slot p % n_win, then unrolled twice —
        # the dual-write ring layout the decode kernel reads (attention.py).
        t = (a[:, -n_win:] if n >= n_win
             else jnp.pad(a, ((0, 0), (n_win - n, 0), (0, 0))))
        t = jnp.roll(t, n % n_win, axis=1)
        return jnp.concatenate([t, t], axis=1)

    cache = cache._replace(swa=jax.tree.map(_ring, swa_q))
    # The in-prefill kernel reads the transient rows (flat layout); pad to
    # n_win so the window DMA never under-runs (pad rows are pos-masked).
    swa_kern = (swa_q if n >= n_win else jax.tree.map(
        lambda a: jnp.pad(a, ((0, 0), (0, n_win - n), (0, 0))), swa_q))

    # --- compressed entries (completed blocks only) → cache ---
    if kind != "swa" and n_full > 0:
        hm = h[:, : n_full * m]
        if kind == "csa":
            kc = eager.csa_compress(hm, p.W_aKV, p.W_bKV, p.W_aZ, p.W_bZ,
                                    p.B_a, p.B_b, m)
            ki = eager.csa_compress(hm, p.W_aIK, p.W_bIK, p.W_aIZ, p.W_bIZ,
                                    p.B_aI, p.B_bI, m)
            kiq = quantize_rows(ki)
            cache = cache._replace(ki=_row_set(cache.ki, kiq, 0))
            P = cfg.csa_pages
            if P and n_full // P > 0:
                # page summaries from the *quantized* rows (what decode
                # recomputes from on page completion — keeps the twins
                # bit-consistent): mean rows, or max/min envelopes in
                # exact mode (§13.7).
                n_pd = n_full // P
                pg_rows = dequantize_rows(kiq)[:, :n_pd * P].reshape(
                    B, n_pd, P, -1)
                if cfg.csa_pages_exact:
                    cache = cache._replace(
                        kis=_row_set(
                            cache.kis,
                            quantize_rows_bound(pg_rows.max(axis=2),
                                                upper=True), 0),
                        kis_lo=_row_set(
                            cache.kis_lo,
                            quantize_rows_bound(pg_rows.min(axis=2),
                                                upper=False), 0))
                else:
                    cache = cache._replace(
                        kis=_row_set(
                            cache.kis, quantize_rows(pg_rows.mean(axis=2)),
                            0))
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
    exact = kind == "csa" and cfg.csa_pages_exact
    paged = kind == "csa" and cfg.csa_pages > 0 and not exact
    if exact:
        # Row-exact f32 scoring over every entry of every COMPLETED page —
        # prefill holds the rows anyway; envelope+rescan bytes only matter
        # for decode. f32 keys (not the legacy bf16 cast) so prefill
        # scores match the decode rescan elementwise, and the same
        # completed-pages reach as decode (the in-progress page sits
        # inside the SWA window either way): one shared twin contract.
        P = cfg.csa_pages
        ki_deq = dequantize_rows(jax.tree.map(
            lambda a: a[:, :max(n_full, 1)], cache.ki))
        scores = eager.lightning_indexer(
            h, ki_deq, p.W_DQ, p.W_IUQ, p.W_w, m, acfg.n_I_h, acfg.c_I)
        s_idx = jnp.arange(ki_deq.shape[1])
        pg_mask = s_idx[None, :] < (t[:, None] // (m * P)) * P
        scores = jnp.where(pg_mask[None], scores, -jnp.inf)
        topk = eager.topk_indices(scores, acfg.topk)      # [B, n, k]
    elif paged:
        # Coarse-to-fine: score only the page summaries (n_blk/P entries).
        # lightning_indexer with block size m*P masks s < t // (m*P) ==
        # (t//m) // P — exactly the completed-pages-only contract.
        P = cfg.csa_pages
        n_pd = n_full // P
        kis_deq = dequantize_rows(jax.tree.map(
            lambda a: a[:, :max(n_pd, 1)], cache.kis))
        cscores = eager.lightning_indexer(
            h, kis_deq.astype(h.dtype),
            p.W_DQ, p.W_IUQ, p.W_w, m * P, acfg.n_I_h, acfg.c_I)
        topk_pg = eager.topk_indices(cscores, acfg.topk // P)  # [B, n, k/P]
    elif kind == "csa":
        ki_deq = dequantize_rows(jax.tree.map(
            lambda a: a[:, :max(n_full, 1)], cache.ki))
        scores = eager.lightning_indexer(
            h, ki_deq.astype(h.dtype),
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
    if paged:
        bound = jnp.broadcast_to((t // m)[None, :], (B, n)).astype(jnp.int32)
        o = sparse_mqa_paged(q, cache.kc, topk_pg, bound, swa_kern, positions,
                             p.attn_sink, page=cfg.csa_pages,
                             n_win=acfg.n_win, rope_dim=rd, tiles=tiles)
    else:
        o = sparse_mqa_gathered(q, cache.kc, topk, swa_kern, positions,
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

    # --- append SWA row (dual-write ring: slot pos % n_win and its twin) ---
    n_win = acfg.n_win
    swa_row = quantize_kv(_swa_rows(h_t, p, rd, positions), rd)  # [B, 1, ...]
    slot = pos % n_win
    swa = _kv_set(cache.swa, swa_row, slot)
    cache = cache._replace(swa=_kv_set(swa, swa_row, slot + n_win))

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
            irow = quantize_rows(ient[:, None, :])
            iold = jax.tree.map(
                lambda a: jax.lax.dynamic_slice(a, (0, at, 0), (B, 1, a.shape[-1])),
                cache.ki)
            cache = cache._replace(ki=_row_set(cache.ki, sel(irow, iold), at))
            P = cfg.csa_pages
            if P:
                # A page completes when its last entry was just written —
                # recompute its summary from the P quantized rows (same
                # inputs the prefill summary used: bit-consistent twins).
                new_page = jnp.logical_and(new_block, (blk_id + 1) % P == 0)
                pg = blk_id // P
                pat = jnp.where(new_page, pg, 0)
                rows = jax.tree.map(
                    lambda a: jax.lax.dynamic_slice(
                        a, (0, jnp.where(new_page, pg * P, 0), 0),
                        (B, P, a.shape[-1])),
                    cache.ki)
                rdeq = dequantize_rows(rows)

                def wr(dst, val, qfn=quantize_rows):
                    srow = qfn(val)
                    sold = jax.tree.map(
                        lambda a: jax.lax.dynamic_slice(
                            a, (0, pat, 0), (B, 1, a.shape[-1])),
                        dst)
                    pick = jax.tree.map(
                        lambda x, y: jnp.where(new_page, x,
                                               y.astype(x.dtype)),
                        srow, sold)
                    return _row_set(dst, pick, pat)

                if cfg.csa_pages_exact:
                    cache = cache._replace(
                        kis=wr(cache.kis, rdeq.max(axis=1, keepdims=True),
                               partial(quantize_rows_bound, upper=True)),
                        kis_lo=wr(cache.kis_lo,
                                  rdeq.min(axis=1, keepdims=True),
                                  partial(quantize_rows_bound, upper=False)))
                else:
                    cache = cache._replace(
                        kis=wr(cache.kis, rdeq.mean(axis=1, keepdims=True)))

    n_blk_valid = (pos + 1) // m                          # completed blocks now

    # --- selection ---
    exact = kind == "csa" and cfg.csa_pages_exact
    paged = kind == "csa" and cfg.csa_pages > 0 and not exact
    if exact:
        # Coarse: sound per-page UBs over the valid envelope prefix.
        # Fine: rescan the top-R pages at row resolution, then row-exact
        # f32 top-k over the candidates (§13.7).
        P = cfg.csa_pages
        n_attend = pos // m
        R = cfg.csa_rescan or 2 * acfg.topk // P
        ub = indexer_ub_scores_decode(h_t, cache.kis, cache.kis_lo, p,
                                      acfg, n_attend // P, tiles=tiles)
        top_pg = eager.topk_indices(ub, R)                # [B, 1, R]
        cscores, rows = _rescan_rows(h_t, cache.ki, p, acfg, top_pg, P,
                                     n_attend)
        topk = _topk_rows(cscores, rows, acfg.topk)       # [B, 1, k]
    elif paged:
        P = cfg.csa_pages
        n_attend = pos // m
        cscores = indexer_scores_decode(h_t, cache.kis, p, acfg,
                                        n_attend // P, tiles=tiles)
        topk_pg = eager.topk_indices(cscores, acfg.topk // P)
    elif kind == "csa":
        n_attend = pos // m                               # eager indexer mask
        scores = indexer_scores_decode(h_t, cache.ki, p, acfg, n_attend,
                                       tiles=tiles)
        topk = eager.topk_indices(scores, acfg.topk)
    elif kind == "hca":
        s_blk = cache.kc.nope.shape[1]
        idx = jnp.arange(s_blk, dtype=jnp.int32)
        topk = jnp.where(idx < n_blk_valid, idx, -1)[None, None, :]
        topk = jnp.broadcast_to(topk, (B, 1, s_blk))
    else:
        topk = jnp.full((B, 1, tiles.attn_chunk), -1, jnp.int32)

    q = _q_proj(h_t, p, n_h, c, rd, positions)
    if paged:
        bound = jnp.full((B, 1), n_attend, jnp.int32)
        o = sparse_mqa_paged(q, cache.kc, topk_pg, bound, cache.swa,
                             positions, p.attn_sink, page=cfg.csa_pages,
                             n_win=acfg.n_win, rope_dim=rd, swa_ring=True,
                             tiles=tiles)
    else:
        o = sparse_mqa_gathered(q, cache.kc, topk, cache.swa, positions,
                                p.attn_sink, n_win=acfg.n_win, rope_dim=rd,
                                swa_ring=True, tiles=tiles)
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


def _mixes_proj(x, mp: MHCParams):
    """Eager mixes projection — the network's FIRST half only. Every later
    half receives its mixes from the previous ``mhc_update_mix`` epilogue
    (HARDWARE_NOTES §13.6), skipping this full hc·d re-read of X."""
    B, n, hc, d = x.shape
    flat = eager.rms_norm(x.astype(jnp.float32)).reshape(B, n, hc * d)
    return flat @ mp.w_mix


def _mhc_gates(mixes, mp: MHCParams, cfg: ModelConfig, tiles: ServingTiles):
    """Sinkhorn projection of precomputed mixes → (pre, post, comb)."""
    kc = KernelConfig(bq=1, bs=128, bn_sinkhorn=tiles.mhc_bn,
                      interpret=tiles.interpret)
    return mhc_sinkhorn_kernel_v2(mixes, mp.scale, mp.base, cfg.hc,
                                  cfg.sinkhorn_iters, 1e-6, config=kc)


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def _block(x, h_fn, mp: MHCParams, cfg, tiles, mixes, w_mix_next):
    """One mHC-wrapped sublayer: X ← comb·X + post⊗F(prenorm(pre·X)).

    ``mixes`` is this half's gate input (threaded from the previous
    half's update epilogue); ``w_mix_next`` is the NEXT half's mixes
    projection, fused into this half's update while X' is in VMEM —
    None for the network's last half (the head reads X' directly).
    Returns ``(X', next mixes | None)``."""
    pre, post, comb = _mhc_gates(mixes, mp, cfg, tiles)
    h = mhc_pre_norm(x, pre, tiles=tiles)
    f = h_fn(h.astype(jnp.bfloat16))
    if w_mix_next is None:
        return mhc_update(x, comb, post, f.astype(x.dtype),
                          tiles=tiles), None
    return mhc_update_mix(x, comb, post, f.astype(x.dtype), w_mix_next,
                          tiles=tiles)


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
    L = len(params.layers)
    mixes = _mixes_proj(x, params.layers[0].mhc_attn)
    for li, lp in enumerate(params.layers):
        def attn_fn(h, _li=li, _lp=lp):
            f, caches[_li] = _attn_prefill(h, _lp, cfg, caches[_li], tiles)
            return f
        x, mixes = _block(x, attn_fn, lp.mhc_attn, cfg, tiles,
                          mixes, lp.mhc_moe.w_mix)
        w_next = (params.layers[li + 1].mhc_attn.w_mix
                  if li + 1 < L else None)
        x, mixes = _block(x, lambda h, _li=li, _lp=lp: _moe_sublayer(
            h, token_ids, _li, _lp, cfg, tiles), lp.mhc_moe, cfg, tiles,
            mixes, w_next)

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
    L = len(params.layers)
    mixes = _mixes_proj(x, params.layers[0].mhc_attn)
    for li, lp in enumerate(params.layers):
        def attn_fn(h, _li=li, _lp=lp):
            f, caches[_li] = _attn_decode(h, pos, _lp, cfg, caches[_li], tiles)
            return f
        x, mixes = _block(x, attn_fn, lp.mhc_attn, cfg, tiles,
                          mixes, lp.mhc_moe.w_mix)
        w_next = (params.layers[li + 1].mhc_attn.w_mix
                  if li + 1 < L else None)
        x, mixes = _block(x, lambda h, _li=li, _lp=lp: _moe_sublayer(
            h, token_id[:, None], _li, _lp, cfg, tiles), lp.mhc_moe, cfg, tiles,
            mixes, w_next)

    h_out = eager.rms_norm(x.astype(jnp.float32).mean(axis=2))
    logits = (h_out @ params.head)[:, 0]
    return logits, ServingState(pos=pos + 1, caches=tuple(caches))
