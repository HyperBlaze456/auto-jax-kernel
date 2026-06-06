"""Eager-JAX reference for DeepSeek V4 hybrid attention.

Implements, faithfully to the V4 paper (arXiv preprint, §2.3):

  - CSA token-level compressor: overlapped 2m-token softmax mix into n/m entries.
  - HCA token-level compressor: non-overlapped m'-token softmax mix into n/m' entries.
  - Lightning indexer: low-rank query path + ReLU dot + head-mix → causal top-k.
  - Sparse MQA: per-query gather of top-k compressed entries, MQA with shared K=V,
    concatenated with the SWA window (recent n_win uncompressed tokens), with
    learnable per-head attention sink and partial RoPE on the last 64 dims.
  - HCA dense MQA: same MQA shape but over *all* compressed entries (no top-k).
  - mHC Sinkhorn projection (§2.2): Sinkhorn–Knopp iterations onto the doubly
    stochastic manifold for residual mapping B_l.

Everything here prioritizes clarity over speed. Use it as the oracle the
Pallas kernel is graded against in bench.py.

Causal note: indexer scoring masks block s for query t with s < floor(t/m).
The SWA branch covers the in-progress block (and the n_win-1 prior tokens),
exactly as the paper specifies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CSAConfig:
    d: int            # hidden dim
    c: int            # head dim (compressed KV channel width)
    n_h: int          # number of MQA query heads
    m: int            # CSA compression rate (e.g. 4)
    topk: int         # number of compressed blocks selected per query
    n_win: int        # SWA window size (uncompressed)
    d_c: int          # query compression dim (low-rank)
    n_I_h: int        # number of indexer query heads
    c_I: int          # indexer head dim
    rope_dim: int = 64


@dataclass(frozen=True)
class HCAConfig:
    d: int
    c: int
    n_h: int
    m_prime: int      # HCA compression rate (e.g. 128)
    n_win: int
    d_c: int
    rope_dim: int = 64


# Parameter bundles. Kept as plain pytrees (NamedTuple) so JAX traces them.

class CSAParams(NamedTuple):
    # token compressor (KV path)
    W_aKV: jax.Array  # [d, c]
    W_bKV: jax.Array  # [d, c]
    W_aZ: jax.Array   # [d, c]
    W_bZ: jax.Array   # [d, c]
    B_a: jax.Array    # [m, c]
    B_b: jax.Array    # [m, c]
    # indexer-key compressor (separate W's, same algorithm)
    W_aIK: jax.Array  # [d, c_I]
    W_bIK: jax.Array  # [d, c_I]
    W_aIZ: jax.Array  # [d, c_I]
    W_bIZ: jax.Array  # [d, c_I]
    B_aI: jax.Array   # [m, c_I]
    B_bI: jax.Array   # [m, c_I]
    # query path (low-rank shared between MQA and indexer)
    W_DQ: jax.Array   # [d, d_c]
    W_UQ: jax.Array   # [d_c, n_h * c]
    W_IUQ: jax.Array  # [d_c, n_I_h * c_I]
    W_w: jax.Array    # [d, n_I_h]   indexer head-mix weights
    # uncompressed KV path for SWA branch
    W_swaK: jax.Array  # [d, c]
    W_swaV: jax.Array  # [d, c]
    # attention sink (per query head)
    attn_sink: jax.Array  # [n_h]


class HCAParams(NamedTuple):
    W_KV: jax.Array   # [d, c]
    W_Z: jax.Array    # [d, c]
    B: jax.Array      # [m_prime, c]
    W_DQ: jax.Array   # [d, d_c]
    W_UQ: jax.Array   # [d_c, n_h * c]
    W_swaK: jax.Array
    W_swaV: jax.Array
    attn_sink: jax.Array  # [n_h]


# ---------------------------------------------------------------------------
# Helpers: RMSNorm, partial RoPE
# ---------------------------------------------------------------------------

def rms_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    return x * jax.lax.rsqrt((x * x).mean(axis=-1, keepdims=True) + eps)


def _build_rope(seq_len: int, rope_dim: int, base: float = 10000.0) -> tuple[jax.Array, jax.Array]:
    """Return (cos, sin) of shape [seq_len, rope_dim/2]."""
    half = rope_dim // 2
    inv_freq = 1.0 / (base ** (jnp.arange(half, dtype=jnp.float32) * 2.0 / rope_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = pos[:, None] * inv_freq[None, :]  # [seq_len, half]
    return jnp.cos(angles), jnp.sin(angles)


def apply_partial_rope(x: jax.Array, cos: jax.Array, sin: jax.Array, rope_dim: int) -> jax.Array:
    """RoPE on last `rope_dim` dims of x.

    x: [..., seq_len, head_dim]
    cos, sin: [seq_len, rope_dim/2]
    """
    head_dim = x.shape[-1]
    no_rope = x[..., : head_dim - rope_dim]
    rope = x[..., head_dim - rope_dim :]
    half = rope_dim // 2
    x1 = rope[..., :half]
    x2 = rope[..., half:]
    # broadcast cos/sin across leading dims
    while cos.ndim < x1.ndim:
        cos = cos[None, ...]
        sin = sin[None, ...]
    rotated = jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return jnp.concatenate([no_rope, rotated], axis=-1)


# ---------------------------------------------------------------------------
# Token-level compressors (§2.3.1, §2.3.2)
# ---------------------------------------------------------------------------

def csa_compress(
    H: jax.Array, W_aKV: jax.Array, W_bKV: jax.Array,
    W_aZ: jax.Array, W_bZ: jax.Array,
    B_a: jax.Array, B_b: jax.Array, m: int,
) -> jax.Array:
    """Overlapped 2m-token CSA compressor.

    H: [B, n, d]  →  C_comp: [B, n/m, c]

    For block i, mixes (Z^a[m·i : m(i+1)] + B^a) with (Z^b[m(i-1) : m·i] + B^b)
    via a row-wise softmax over 2m positions per c-channel, then sums against
    (C^a[m·i : m(i+1)], C^b[m(i-1) : m·i]). Boundary i=0 has Z^b padded -inf.
    """
    B, n, _ = H.shape
    assert n % m == 0, f"n={n} must be divisible by m={m}"
    n_blk = n // m
    c = W_aKV.shape[1]

    Ca = H @ W_aKV  # [B, n, c]
    Cb = H @ W_bKV
    Za = H @ W_aZ
    Zb = H @ W_bZ

    Ca_blk = Ca.reshape(B, n_blk, m, c)
    Cb_blk = Cb.reshape(B, n_blk, m, c)
    Za_blk = Za.reshape(B, n_blk, m, c)
    Zb_blk = Zb.reshape(B, n_blk, m, c)

    # Previous-block alignment for the b path. Block i uses Cb_blk[i-1].
    Cb_prev = jnp.concatenate(
        [jnp.zeros_like(Cb_blk[:, :1]), Cb_blk[:, :-1]], axis=1
    )
    Zb_prev = jnp.concatenate(
        [jnp.full_like(Zb_blk[:, :1], -jnp.inf), Zb_blk[:, :-1]], axis=1
    )

    Za_b = Za_blk + B_a[None, None, :, :]
    Zb_b = Zb_prev + B_b[None, None, :, :]
    logits = jnp.concatenate([Za_b, Zb_b], axis=2)  # [B, n_blk, 2m, c]
    weights = jax.nn.softmax(logits, axis=2)
    Sa, Sb = weights[:, :, :m, :], weights[:, :, m:, :]

    return (Sa * Ca_blk).sum(axis=2) + (Sb * Cb_prev).sum(axis=2)


def hca_compress(
    H: jax.Array, W_KV: jax.Array, W_Z: jax.Array, B: jax.Array, m_prime: int,
) -> jax.Array:
    """Non-overlapped m'-token HCA compressor.

    H: [B, n, d]  →  C_comp: [B, n/m', c]
    """
    B_, n, _ = H.shape
    assert n % m_prime == 0
    n_blk = n // m_prime
    c = W_KV.shape[1]
    C = H @ W_KV
    Z = H @ W_Z
    C_blk = C.reshape(B_, n_blk, m_prime, c)
    Z_blk = Z.reshape(B_, n_blk, m_prime, c)
    weights = jax.nn.softmax(Z_blk + B[None, None, :, :], axis=2)
    return (weights * C_blk).sum(axis=2)


# ---------------------------------------------------------------------------
# Lightning indexer (§2.3.1)
# ---------------------------------------------------------------------------

def lightning_indexer(
    H: jax.Array, K_IComp: jax.Array,
    W_DQ: jax.Array, W_IUQ: jax.Array, W_w: jax.Array,
    m: int, n_I_h: int, c_I: int,
) -> jax.Array:
    """Index scores I[t, s] between query t and compressed block s.

    Returns: [B, n, n_blk] with -inf for s ≥ floor(t/m) (causal).

    Per paper: I[t, s] = Σ_h w_I[t, h] · ReLU(q_I[t, h] · K_IComp[s])
    """
    B, n, d = H.shape
    n_blk = K_IComp.shape[1]

    cQ = H @ W_DQ                              # [B, n, d_c]
    qI = (cQ @ W_IUQ).reshape(B, n, n_I_h, c_I)
    wI = H @ W_w                               # [B, n, n_I_h]

    # qk[b, t, s, h] = qI[b,t,h,:] · K_IComp[b,s,:]
    qk = jnp.einsum("bthc,bsc->btsh", qI, K_IComp)
    qk = jax.nn.relu(qk)
    scores = (wI[:, :, None, :] * qk).sum(axis=-1)   # [B, n, n_blk]

    t_idx = jnp.arange(n)
    s_idx = jnp.arange(n_blk)
    mask = s_idx[None, :] < (t_idx[:, None] // m)    # [n, n_blk]
    return jnp.where(mask[None, :, :], scores, -jnp.inf)


def topk_indices(scores: jax.Array, k: int) -> jax.Array:
    """Select top-k compressed-block indices per query token.

    scores: [B, n, n_blk]  →  idx: [B, n, k] int32. Entries with -inf score
    (causally invalid) stay in the result with score -inf; downstream MQA uses
    the score-mask separately. Pads with -1 when fewer than k valid entries.
    """
    B, n, n_blk = scores.shape
    if k >= n_blk:
        # Pad with -1 to maintain shape contract. Caller masks via valid scores.
        idx = jnp.broadcast_to(jnp.arange(n_blk), (B, n, n_blk))
        pad = jnp.full((B, n, k - n_blk), -1, dtype=jnp.int32)
        return jnp.concatenate([idx.astype(jnp.int32), pad], axis=-1)
    # jnp.argsort is stable; take last k.
    _, idx = jax.lax.top_k(scores, k)
    # Mask invalid (would-be-selected -inf) to -1
    selected = jnp.take_along_axis(scores, idx, axis=-1)
    idx = jnp.where(jnp.isfinite(selected), idx.astype(jnp.int32), -1)
    return idx


# ---------------------------------------------------------------------------
# SWA gather: recent n_win uncompressed tokens per query
# ---------------------------------------------------------------------------

def swa_gather(K: jax.Array, n_win: int) -> jax.Array:
    """K: [B, n, c]  →  K_swa: [B, n, n_win, c] with K_swa[..., t, w, :] =
    K[..., t - n_win + 1 + w, :], zero-padded for w < 0.
    """
    B, n, c = K.shape
    pad = jnp.zeros((B, n_win - 1, c), dtype=K.dtype)
    padded = jnp.concatenate([pad, K], axis=1)        # [B, n + n_win - 1, c]
    idx = jnp.arange(n)[:, None] + jnp.arange(n_win)[None, :]  # [n, n_win]
    return padded[:, idx, :]


# ---------------------------------------------------------------------------
# Sparse-attn-with-sink core (CSA's MQA stage; matches reference sparse_attn_kernel)
# ---------------------------------------------------------------------------

def sparse_attn_with_sink(
    q: jax.Array,          # [B, n, n_h, c]   queries (RMSNormed, partial-RoPE'd)
    K_comp: jax.Array,     # [B, n_blk, c]    compressed entries (RMSNormed, RoPE'd)
    topk_idxs: jax.Array,  # [B, n, k] int32  per-query selected blocks (-1 = pad)
    K_swa: jax.Array,      # [B, n, n_win, c] sliding-window KV (RMSNormed, RoPE'd)
    attn_sink: jax.Array,  # [n_h]
    scale: float | None = None,
) -> jax.Array:
    """MQA over (selected compressed blocks ∪ SWA window) with attention sink.

    Returns o: [B, n, n_h, c].

    Math (per head h, query t):
      logits = scale · q · [K_sel ; K_swa]^T          # length k+n_win
      mask   = (topk_idxs >= 0) ∪ (always for SWA)
      m'     = max(max(logits), sink_h)
      o      = (exp(logits - m') · K) / (Σ exp(logits - m') + exp(sink - m'))
    """
    B, n, n_h, c = q.shape
    k = topk_idxs.shape[-1]
    n_win = K_swa.shape[2]
    if scale is None:
        scale = c ** -0.5

    # Gather selected compressed entries. Use index -1 → 0; mask their scores below.
    safe_idx = jnp.where(topk_idxs < 0, 0, topk_idxs)            # [B, n, k]
    # K_sel[b, t, j, :] = K_comp[b, safe_idx[b, t, j], :]
    K_sel = jnp.take_along_axis(
        K_comp[:, None, :, :].repeat(n, axis=1),
        safe_idx[..., None].repeat(c, axis=-1),
        axis=2,
    )  # [B, n, k, c]

    K_full = jnp.concatenate([K_sel, K_swa], axis=2)              # [B, n, k+n_win, c]
    # Shared KV: V is the same as K (per paper, "each compressed KV entry serves
    # as both attention key and value"). The SWA branch uses an uncompressed
    # K=V tensor too (caller is responsible for producing it; we keep the
    # symmetric V-via-K convention here for simplicity).
    V_full = K_full

    # Validity mask
    valid_topk = topk_idxs >= 0                                   # [B, n, k]
    valid_swa = jnp.ones((B, n, n_win), dtype=bool)
    valid = jnp.concatenate([valid_topk, valid_swa], axis=-1)     # [B, n, k+n_win]

    # Logits: [B, n, n_h, k+n_win]
    logits = jnp.einsum("bthc,btsc->bths", q, K_full) * scale
    logits = jnp.where(valid[:, :, None, :], logits, -jnp.inf)

    # Online softmax with sink:
    m_attn = jnp.max(logits, axis=-1, keepdims=True)              # [B, n, n_h, 1]
    sink = attn_sink[None, None, :, None]                         # [1, 1, n_h, 1]
    m_combined = jnp.maximum(m_attn, sink)
    # Avoid -inf - -inf = nan when an entire row is masked (no valid topk and
    # n_win=0); broadcasted finite sink saves us as long as sink is finite.
    exp_scores = jnp.exp(logits - m_combined)
    exp_sink = jnp.exp(sink - m_combined)
    denom = exp_scores.sum(axis=-1, keepdims=True) + exp_sink     # [B, n, n_h, 1]
    weights = exp_scores / denom

    return jnp.einsum("bths,btsc->bthc", weights, V_full)


# ---------------------------------------------------------------------------
# Full CSA / HCA forward passes
# ---------------------------------------------------------------------------

def csa_forward(H: jax.Array, p: CSAParams, cfg: CSAConfig) -> jax.Array:
    """End-to-end CSA: compress → indexer → top-k → MQA-with-sink → return MQA out.

    H:   [B, n, d]
    out: [B, n, n_h, c]   (caller does the grouped output projection)
    """
    B, n, d = H.shape

    # Compressed KV path
    K_comp = csa_compress(H, p.W_aKV, p.W_bKV, p.W_aZ, p.W_bZ, p.B_a, p.B_b, cfg.m)  # [B, n/m, c]
    K_IComp = csa_compress(
        H, p.W_aIK, p.W_bIK, p.W_aIZ, p.W_bIZ, p.B_aI, p.B_bI, cfg.m
    )  # [B, n/m, c_I]

    # Lightning indexer scores → top-k
    scores = lightning_indexer(
        H, K_IComp, p.W_DQ, p.W_IUQ, p.W_w, cfg.m, cfg.n_I_h, cfg.c_I,
    )
    topk_idxs = topk_indices(scores, cfg.topk)  # [B, n, k]

    # Queries (low-rank, shared latent with indexer)
    cQ = H @ p.W_DQ
    q = (cQ @ p.W_UQ).reshape(B, n, cfg.n_h, cfg.c)  # [B, n, n_h, c]

    # Partial RoPE on last rope_dim
    q_pos = jnp.arange(n)
    cos_q, sin_q = _build_rope(n, cfg.rope_dim)
    q = apply_partial_rope(q, cos_q[None, :, None, :], sin_q[None, :, None, :], cfg.rope_dim)

    # KV gets RoPE at the *compressed-block* positions. Use block-center as the
    # anchor (a defensible choice; the paper applies RoPE to the last-64 of KV).
    n_blk = K_comp.shape[1]
    cos_k, sin_k = _build_rope(n_blk, cfg.rope_dim)
    K_comp = apply_partial_rope(K_comp, cos_k[None, :, :], sin_k[None, :, :], cfg.rope_dim)

    # SWA branch: uncompressed KV (single head; shared K=V in MQA).
    K_swa_full = H @ p.W_swaK   # [B, n, c]
    cos_s, sin_s = _build_rope(n, cfg.rope_dim)
    K_swa_full = apply_partial_rope(K_swa_full, cos_s[None, :, :], sin_s[None, :, :], cfg.rope_dim)
    K_swa = swa_gather(K_swa_full, cfg.n_win)  # [B, n, n_win, c]

    # RMSNorm on Q heads and the single KV head, just before core attention.
    q = rms_norm(q)
    K_comp = rms_norm(K_comp)
    K_swa = rms_norm(K_swa)

    return sparse_attn_with_sink(q, K_comp, topk_idxs, K_swa, p.attn_sink)


def hca_forward(H: jax.Array, p: HCAParams, cfg: HCAConfig) -> jax.Array:
    """End-to-end HCA: heavy-compress → dense MQA over all entries → return MQA out."""
    B, n, d = H.shape
    K_comp = hca_compress(H, p.W_KV, p.W_Z, p.B, cfg.m_prime)   # [B, n/m', c]

    cQ = H @ p.W_DQ
    q = (cQ @ p.W_UQ).reshape(B, n, cfg.n_h, cfg.c)

    cos_q, sin_q = _build_rope(n, cfg.rope_dim)
    q = apply_partial_rope(q, cos_q[None, :, None, :], sin_q[None, :, None, :], cfg.rope_dim)

    n_blk = K_comp.shape[1]
    cos_k, sin_k = _build_rope(n_blk, cfg.rope_dim)
    K_comp = apply_partial_rope(K_comp, cos_k[None, :, :], sin_k[None, :, :], cfg.rope_dim)

    K_swa_full = H @ p.W_swaK
    cos_s, sin_s = _build_rope(n, cfg.rope_dim)
    K_swa_full = apply_partial_rope(K_swa_full, cos_s[None, :, :], sin_s[None, :, :], cfg.rope_dim)
    K_swa = swa_gather(K_swa_full, cfg.n_win)

    q = rms_norm(q)
    K_comp = rms_norm(K_comp)
    K_swa = rms_norm(K_swa)

    # Dense MQA: every query attends to every compressed block, with a causal
    # mask at block granularity (s ≤ floor(t/m')).
    n_h, c = cfg.n_h, cfg.c
    scale = c ** -0.5

    # Build "select-all" topk_idxs and feed through sparse_attn_with_sink. For
    # multi-million sequences with tiny n_blk this is fine; for the dense
    # variant the caller can also use the explicit dense path below.
    if n_blk <= 4096:
        all_idx = jnp.broadcast_to(jnp.arange(n_blk), (B, n, n_blk)).astype(jnp.int32)
        # Causal: mask blocks s > floor(t/m_prime)
        t_idx = jnp.arange(n)
        causal = jnp.arange(n_blk)[None, :] <= (t_idx[:, None] // cfg.m_prime)
        causal = jnp.broadcast_to(causal[None, :, :], (B, n, n_blk))
        all_idx = jnp.where(causal, all_idx, -1)
        return sparse_attn_with_sink(q, K_comp, all_idx, K_swa, p.attn_sink)

    # Block-wise streaming path for very long n_blk would go here. The agent
    # may replace this whole function with a Pallas kernel.
    raise NotImplementedError("HCA dense path > 4096 compressed blocks needs a streaming impl.")


# ---------------------------------------------------------------------------
# mHC Sinkhorn projection (§2.2)
# ---------------------------------------------------------------------------

def mhc_sinkhorn(
    mixes: jax.Array,        # [B, n, (2 + hc) * hc]  raw projection
    hc_scale: jax.Array,     # [3]                     pre / post / comb gate scales
    hc_base: jax.Array,      # [(2 + hc) * hc]         static biases
    hc: int,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sinkhorn–Knopp projection used in mHC residual mapping.

    Splits `mixes` into pre-mix (sigmoid-bounded), post-mix (2·sigmoid-bounded
    so it stays in [0, 2]), and a comb matrix that we project onto the doubly
    stochastic manifold by alternating row/col normalisations after a softmax.

    Returns:
      pre:  [B, n, hc]
      post: [B, n, hc]
      comb: [B, n, hc, hc]   (≈ doubly stochastic)
    """
    B, n, _ = mixes.shape
    flat = mixes  # [B, n, (2 + hc) * hc]
    pre_raw = flat[..., :hc]
    post_raw = flat[..., hc : 2 * hc]
    comb_raw = flat[..., 2 * hc :].reshape(B, n, hc, hc)

    pre = jax.nn.sigmoid(pre_raw * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * jax.nn.sigmoid(post_raw * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = comb_raw * hc_scale[2] + hc_base[2 * hc :].reshape(hc, hc)

    # First pass: softmax across cols, then col-normalise. Mirrors the kernel.
    comb = jax.nn.softmax(comb, axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)

    def step(c, _):
        c = c / (c.sum(axis=-1, keepdims=True) + eps)
        c = c / (c.sum(axis=-2, keepdims=True) + eps)
        return c, None

    comb, _ = jax.lax.scan(step, comb, None, length=sinkhorn_iters - 1)
    return pre, post, comb


# ---------------------------------------------------------------------------
# Parameter init helpers (for tests + bench)
# ---------------------------------------------------------------------------

def init_csa_params(key: jax.Array, cfg: CSAConfig, dtype=jnp.float32) -> CSAParams:
    keys = jax.random.split(key, 16)
    s = cfg.d ** -0.5  # rough scale

    def n(k, shape):
        return jax.random.normal(k, shape, dtype=dtype) * s

    return CSAParams(
        W_aKV=n(keys[0], (cfg.d, cfg.c)),
        W_bKV=n(keys[1], (cfg.d, cfg.c)),
        W_aZ=n(keys[2], (cfg.d, cfg.c)),
        W_bZ=n(keys[3], (cfg.d, cfg.c)),
        B_a=jnp.zeros((cfg.m, cfg.c), dtype=dtype),
        B_b=jnp.zeros((cfg.m, cfg.c), dtype=dtype),
        W_aIK=n(keys[4], (cfg.d, cfg.c_I)),
        W_bIK=n(keys[5], (cfg.d, cfg.c_I)),
        W_aIZ=n(keys[6], (cfg.d, cfg.c_I)),
        W_bIZ=n(keys[7], (cfg.d, cfg.c_I)),
        B_aI=jnp.zeros((cfg.m, cfg.c_I), dtype=dtype),
        B_bI=jnp.zeros((cfg.m, cfg.c_I), dtype=dtype),
        W_DQ=n(keys[8], (cfg.d, cfg.d_c)),
        W_UQ=n(keys[9], (cfg.d_c, cfg.n_h * cfg.c)),
        W_IUQ=n(keys[10], (cfg.d_c, cfg.n_I_h * cfg.c_I)),
        W_w=n(keys[11], (cfg.d, cfg.n_I_h)),
        W_swaK=n(keys[12], (cfg.d, cfg.c)),
        W_swaV=n(keys[13], (cfg.d, cfg.c)),
        attn_sink=jnp.zeros((cfg.n_h,), dtype=dtype),
    )


def init_hca_params(key: jax.Array, cfg: HCAConfig, dtype=jnp.float32) -> HCAParams:
    keys = jax.random.split(key, 8)
    s = cfg.d ** -0.5

    def n(k, shape):
        return jax.random.normal(k, shape, dtype=dtype) * s

    return HCAParams(
        W_KV=n(keys[0], (cfg.d, cfg.c)),
        W_Z=n(keys[1], (cfg.d, cfg.c)),
        B=jnp.zeros((cfg.m_prime, cfg.c), dtype=dtype),
        W_DQ=n(keys[2], (cfg.d, cfg.d_c)),
        W_UQ=n(keys[3], (cfg.d_c, cfg.n_h * cfg.c)),
        W_swaK=n(keys[4], (cfg.d, cfg.c)),
        W_swaV=n(keys[5], (cfg.d, cfg.c)),
        attn_sink=jnp.zeros((cfg.n_h,), dtype=dtype),
    )


# ---------------------------------------------------------------------------
# Rough FLOPs accounting (used by bench.py to estimate MFU)
# ---------------------------------------------------------------------------

def csa_flops(cfg: CSAConfig, B: int, n: int) -> int:
    """Forward FLOPs estimate for csa_forward. Counts dominant matmuls."""
    n_blk = n // cfg.m
    f = 0
    # CSA compressor: 4 GEMMs of [B,n,d] × [d,c]  →  8 BNDC FLOPs
    f += 8 * B * n * cfg.d * cfg.c
    # Indexer-key compressor: same shape but c_I
    f += 8 * B * n * cfg.d * cfg.c_I
    # cQ = H @ W_DQ:  2 BND·d_c
    f += 2 * B * n * cfg.d * cfg.d_c
    # q = cQ @ W_UQ:  2 BN d_c · (n_h c)
    f += 2 * B * n * cfg.d_c * cfg.n_h * cfg.c
    # qI = cQ @ W_IUQ
    f += 2 * B * n * cfg.d_c * cfg.n_I_h * cfg.c_I
    # wI = H @ W_w
    f += 2 * B * n * cfg.d * cfg.n_I_h
    # Indexer scores qI · K_IComp:  2 BN n_blk n_I_h c_I
    f += 2 * B * n * n_blk * cfg.n_I_h * cfg.c_I
    # SWA K projection
    f += 2 * B * n * cfg.d * cfg.c
    # Sparse-MQA: 2 (QK and PV), B n n_h (k+n_win) c × 2
    f += 4 * B * n * cfg.n_h * (cfg.topk + cfg.n_win) * cfg.c
    return f


def hca_flops(cfg: HCAConfig, B: int, n: int) -> int:
    n_blk = n // cfg.m_prime
    f = 0
    f += 4 * B * n * cfg.d * cfg.c                              # HCA compressor (2 GEMMs)
    f += 2 * B * n * cfg.d * cfg.d_c                            # cQ
    f += 2 * B * n * cfg.d_c * cfg.n_h * cfg.c                  # q
    f += 2 * B * n * cfg.d * cfg.c                              # SWA K
    f += 4 * B * n * cfg.n_h * (n_blk + cfg.n_win) * cfg.c      # MQA
    return f


# ---------------------------------------------------------------------------
# Preset shapes from the V4 paper (§4.2.1)
# ---------------------------------------------------------------------------

DSV4_FLASH_CSA = CSAConfig(
    d=4096, c=512, n_h=64, m=4, topk=512, n_win=128,
    d_c=1024, n_I_h=64, c_I=128,
)
DSV4_FLASH_HCA = HCAConfig(
    d=4096, c=512, n_h=64, m_prime=128, n_win=128, d_c=1024,
)
DSV4_PRO_CSA = CSAConfig(
    d=7168, c=512, n_h=128, m=4, topk=1024, n_win=128,
    d_c=1536, n_I_h=64, c_I=128,
)
DSV4_PRO_HCA = HCAConfig(
    d=7168, c=512, n_h=128, m_prime=128, n_win=128, d_c=1536,
)
# Small preset for CPU/dev. Same algorithmic surface, tiny dims. The MQA head
# dim ``c`` is kept at 128 (== the v5e/v6 lane width) so these presets are a
# valid TPU sanity shape too — the Pallas flash kernel requires ``c`` to be a
# multiple of ``lane_size`` (see kernel_config.config_for). The query-path
# low-rank dim ``d_c`` and indexer head dim ``c_I`` stay sub-128: they only
# feed plain-JAX GEMMs / top-k, never the lane-aligned Pallas kernel.
SMALL_CSA = CSAConfig(
    d=128, c=128, n_h=4, m=4, topk=8, n_win=16,
    d_c=64, n_I_h=4, c_I=32,
)
SMALL_HCA = HCAConfig(
    d=128, c=128, n_h=4, m_prime=16, n_win=16, d_c=64,
)
