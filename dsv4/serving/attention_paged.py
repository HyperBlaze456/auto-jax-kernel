"""Page-aligned top-k selection + paged gather attention (HARDWARE_NOTES §9.2).

Why pages
---------

The row-gather kernel (``attention.py``) issues one DMA descriptor per
selected compressed entry: 448–1024 B copies, 3·k descriptors per token.
Each TPU DMA engine sustains only O(10⁷–10⁸) descriptors/s — at k=512
rows/token the gather is **descriptor-issue-bound**, not bandwidth-bound:
the engine idles between tiny transfers.

Constraining selection to *pages* of P consecutive compressed entries
(FlashMLA's paged-KV idea applied to the indexer) fixes this at the
model-contract level:

  - one descriptor moves a contiguous ``P · row_bytes`` slab
    (P=8, hybrid format → ~4.6 KB: comfortably in the DMA engine's
    efficient regime);
  - descriptor count drops P×: 3·(k/P) per token (k=512, P=8 → 192 vs
    1536);
  - HBM row activations improve too — pages are physically contiguous,
    so the read pattern is P-sequential instead of random-row.

Selection contract
------------------

``topk_pages`` scores each page as the **max** of its rows' indexer
scores (max, not sum: one strong row must be able to pull in its page;
sum would bias toward uniformly-mediocre pages) and takes the top
``k_pages = k // P``. The attended-row *budget* is unchanged; the
attended *set* differs from row-top-k — selection coarsening is a model
quality knob, the same trade FlashMLA-style paged sparse attention makes.
``page_recall`` measures the overlap (the paper reports 99.7% recall for
its own selector coarsening, BF16 scores; ours is structural).

Causality is enforced *inside the kernel*, not by the selector: a
selected page may straddle the per-token causal boundary, so the kernel
receives the row bound (``n_valid = t // m`` for CSA) as an SMEM scalar
and masks ``page·P + i >= bound`` rows. The selector only guarantees a
page has ≥1 causally-valid row (else index −1).

Equivalence oracle: a paged selection expanded to row indices and fed to
the *row* path must match this kernel exactly — that test pins the kernel
against already-verified machinery, isolating "pages" as the only new
variable.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from .config import ServingTiles
from .quant import KVQuant

# ---------------------------------------------------------------------------
# Page-aligned selection
# ---------------------------------------------------------------------------


def topk_pages(
    scores: jax.Array,     # [B, n, n_blk] (-inf at causally-invalid rows)
    k_pages: int,
    page: int,
) -> jax.Array:
    """Top-``k_pages`` page indices per query token; -1 where fewer than
    ``k_pages`` pages have any valid row. ``n_blk`` is padded up to a page
    multiple internally (pad rows score -inf)."""
    B, n, n_blk = scores.shape
    pad = (-n_blk) % page
    if pad:
        scores = jnp.pad(scores, ((0, 0), (0, 0), (0, pad)),
                         constant_values=-jnp.inf)
    n_pages = (n_blk + pad) // page
    pscores = scores.reshape(B, n, n_pages, page).max(axis=-1)
    if k_pages >= n_pages:
        idx = jnp.broadcast_to(jnp.arange(n_pages, dtype=jnp.int32),
                               (B, n, n_pages))
        idx = jnp.where(jnp.isfinite(pscores), idx, -1)
        fill = jnp.full((B, n, k_pages - n_pages), -1, jnp.int32)
        return jnp.concatenate([idx, fill], axis=-1)
    top, idx = jax.lax.top_k(pscores, k_pages)
    return jnp.where(jnp.isfinite(top), idx.astype(jnp.int32), -1)


def expand_pages_to_rows(page_idx: jax.Array, page: int,
                         bound: jax.Array) -> jax.Array:
    """[B, T, kp] page indices → [B, T, kp·P] row indices with −1 at
    invalid rows (page −1, or row ≥ per-token causal ``bound`` [B, T]).
    This is the bridge to the row-path oracle."""
    rows = page_idx[..., None] * page + jnp.arange(page, dtype=jnp.int32)
    valid = (page_idx[..., None] >= 0) & (rows < bound[..., None, None])
    rows = jnp.where(valid, rows, -1)
    return rows.reshape(*page_idx.shape[:-1], -1)


def page_recall(row_topk: jax.Array, page_idx: jax.Array, page: int,
                bound: jax.Array) -> jax.Array:
    """Fraction of row-top-k selected rows covered by the paged selection
    (per token, averaged over valid rows). Diagnostic for the selection-
    coarsening quality trade."""
    rows_paged = expand_pages_to_rows(page_idx, page, bound)      # [B,T,kp*P]
    hit = (row_topk[..., :, None] == rows_paged[..., None, :]).any(-1)
    valid = row_topk >= 0
    return (hit & valid).sum() / jnp.maximum(valid.sum(), 1)


# ---------------------------------------------------------------------------
# Paged gather kernel
# ---------------------------------------------------------------------------


def _paged_chunk_copies(idx_smem, base, pages, page, slot,
                        nope_hbm, rope_hbm, scale_hbm,
                        nope_buf, rope_buf, scale_buf, sem, b):
    """One DMA descriptor per page per component — the entire point."""
    copies = []
    for i in range(pages):
        pg = jnp.maximum(idx_smem[0, 0, base + i], 0)
        row0 = pg * page
        copies.append(pltpu.make_async_copy(
            nope_hbm.at[b, pl.ds(row0, page)],
            nope_buf.at[slot, pl.ds(i * page, page)], sem))
        copies.append(pltpu.make_async_copy(
            rope_hbm.at[b, pl.ds(row0, page)],
            rope_buf.at[slot, pl.ds(i * page, page)], sem))
        copies.append(pltpu.make_async_copy(
            scale_hbm.at[b, pl.ds(row0, page)],
            scale_buf.at[slot, pl.ds(i * page, page)], sem))
    return copies


def _paged_mqa_kernel(
    idx_smem_ref,      # [1, 1, kp_pad] int32 (SMEM)  page indices
    pos_smem_ref,      # [1, 1] int32 (SMEM)          query position
    bound_smem_ref,    # [1, 1] int32 (SMEM)          causal row bound
    q_nope_ref,        # [1, 1, n_h, c_nope] bf16
    q_rope_ref,        # [1, 1, n_h, r] bf16
    idx_vmem_ref,      # [1, 1, kp_pad] int32
    sink_ref,          # [n_h, 1] f32
    kc_nope_hbm, kc_rope_hbm, kc_scale_hbm,        # [B, S_c, ...] (ANY)
    sw_nope_hbm, sw_rope_hbm, sw_scale_hbm,        # [B, S_r, ...] (ANY)
    o_nope_ref,        # [1, 1, n_h, c_nope] bf16
    o_rope_ref,        # [1, 1, n_h, r] bf16
    nope_buf,          # [2, pages*P, c_nope]
    rope_buf,          # [2, pages*P, r]
    scale_buf,         # [2, pages*P, 1]
    swn_buf, swr_buf, sws_buf,                     # [n_win, ...]
    gather_sem, swa_sem,
    *,
    kp_pad: int, pages: int, page: int, n_win: int, s_raw: int, scale: float,
    phi_shift: float, swa_ring: bool,
):
    b = pl.program_id(0)
    n_chunks = kp_pad // pages
    pos = pos_smem_ref[0, 0]
    bound = bound_smem_ref[0, 0]
    rows = pages * page

    if swa_ring:
        # Dual-write unrolled ring (s_raw == 2*n_win) — see attention.py.
        start = (pos + 1) % n_win
    else:
        start = jnp.clip(pos - n_win + 1, 0, s_raw - n_win)

    def swa_copies():
        return [
            pltpu.make_async_copy(sw_nope_hbm.at[b, pl.ds(start, n_win)], swn_buf, swa_sem),
            pltpu.make_async_copy(sw_rope_hbm.at[b, pl.ds(start, n_win)], swr_buf, swa_sem),
            pltpu.make_async_copy(sw_scale_hbm.at[b, pl.ds(start, n_win)], sws_buf, swa_sem),
        ]
    for c in swa_copies():
        c.start()

    mk = partial(_paged_chunk_copies, idx_smem_ref, pages=pages, page=page,
                 nope_hbm=kc_nope_hbm, rope_hbm=kc_rope_hbm,
                 scale_hbm=kc_scale_hbm, nope_buf=nope_buf,
                 rope_buf=rope_buf, scale_buf=scale_buf, sem=gather_sem, b=b)
    for c in mk(0, slot=0):
        c.start()

    q_nope = q_nope_ref[0, 0].astype(jnp.bfloat16)
    q_rope = q_rope_ref[0, 0].astype(jnp.bfloat16)
    n_h = q_nope.shape[0]
    l_i = jnp.zeros((n_h, 1), jnp.float32)
    acc_n = jnp.zeros(o_nope_ref.shape[2:], jnp.float32)
    acc_r = jnp.zeros(o_rope_ref.shape[2:], jnp.float32)

    def phi_update(l_i, acc_n, acc_r, k_n, k_r, valid):
        # Decoupled phi-softmax: constant shift C=phi_shift (=sqrt(c)) for the
        # running max — see attention.py. Pure accumulators, no acc*alpha.
        logits = (
            jax.lax.dot_general(q_nope, k_n, (((1,), (1,)), ((), ())),
                                preferred_element_type=jnp.float32)
            + jax.lax.dot_general(q_rope, k_r, (((1,), (1,)), ((), ())),
                                  preferred_element_type=jnp.float32)
        ) * jnp.float32(scale)
        phi = jnp.where(valid, jnp.exp(logits - jnp.float32(phi_shift)), 0.0)
        l_new = l_i + phi.sum(-1, keepdims=True)
        phi_bf = phi.astype(jnp.bfloat16)
        acc_n = acc_n + jax.lax.dot_general(
            phi_bf, k_n, (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32)
        acc_r = acc_r + jax.lax.dot_general(
            phi_bf, k_r, (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32)
        return l_new, acc_n, acc_r

    for j in range(n_chunks):
        slot = j % 2
        if j + 1 < n_chunks:
            for c in mk((j + 1) * pages, slot=1 - slot):
                c.start()
        for c in mk(j * pages, slot=slot):
            c.wait()

        k_n = (nope_buf[slot].astype(jnp.float32)
               * scale_buf[slot]).astype(jnp.bfloat16)       # [rows, c_nope]
        k_r = rope_buf[slot].astype(jnp.bfloat16)
        # Row validity: page valid AND global row < causal bound.
        pg = idx_vmem_ref[0, 0, j * pages:(j + 1) * pages]    # [pages]
        within = jax.lax.broadcasted_iota(jnp.int32, (pages, page), 1)
        gl_rows = pg[:, None] * page + within                 # [pages, P]
        valid = ((pg[:, None] >= 0) & (gl_rows < bound)).reshape(1, rows)
        l_i, acc_n, acc_r = phi_update(l_i, acc_n, acc_r, k_n, k_r, valid)

    for c in swa_copies():
        c.wait()
    k_n = (swn_buf[...].astype(jnp.float32) * sws_buf[...]).astype(jnp.bfloat16)
    k_r = swr_buf[...].astype(jnp.bfloat16)
    if swa_ring:
        w_pos = (pos - n_win + 1
                 + jax.lax.broadcasted_iota(jnp.int32, (1, n_win), 1))
        valid = w_pos >= 0
    else:
        w_pos = start + jax.lax.broadcasted_iota(jnp.int32, (1, n_win), 1)
        valid = jnp.logical_and(w_pos <= pos, w_pos > pos - n_win)
    l_i, acc_n, acc_r = phi_update(l_i, acc_n, acc_r, k_n, k_r, valid)

    sink = sink_ref[...]
    denom = l_i + jnp.exp(sink - jnp.float32(phi_shift))
    o_nope_ref[0, 0] = (acc_n / denom).astype(o_nope_ref.dtype)
    o_rope_ref[0, 0] = (acc_r / denom).astype(o_rope_ref.dtype)


def sparse_mqa_paged(
    q: jax.Array,            # [B, T, n_h, c]
    kc: KVQuant,             # compressed cache (S_c a multiple of page is
                             # NOT required: bound masking covers the tail,
                             # but S_c must be >= n_pages*page rows of
                             # allocated storage — callers size caches in
                             # page multiples)
    page_idx: jax.Array,     # [B, T, kp] int32 page indices, -1 invalid
    bound: jax.Array,        # [B, T] int32 causal row bound (rows < bound)
    swa: KVQuant,
    q_pos: jax.Array,        # [B, T]
    attn_sink: jax.Array,    # [n_h]
    *,
    page: int,
    n_win: int,
    rope_dim: int = 64,
    swa_ring: bool = False,
    tiles: ServingTiles | None = None,
) -> jax.Array:
    """Paged-gather sparse MQA. Same output contract as
    ``attention.sparse_mqa_gathered`` over ``expand_pages_to_rows`` —
    asserted by the equivalence test. ``swa_ring`` as in attention.py."""
    if tiles is None:
        from .config import tiles_for
        tiles = tiles_for()
    B, T, n_h, c = q.shape
    c_nope = c - rope_dim
    kp = page_idx.shape[-1]
    s_raw = swa.nope.shape[1]
    s_c = kc.nope.shape[1]
    if swa_ring and s_raw != 2 * n_win:
        raise ValueError(f"ring SWA cache must be 2*n_win={2*n_win} rows, "
                         f"got {s_raw}")
    if s_c % page:
        raise ValueError(f"compressed cache length {s_c} must be a multiple "
                         f"of page={page} (allocate in page multiples)")

    # Pages fetched per DMA wave: keep the per-wave row count near the row
    # kernel's chunk so VMEM stays comparable.
    pages = max(1, tiles.attn_chunk // page)
    kp_pad = ((kp + pages - 1) // pages) * pages
    if kp_pad != kp:
        page_idx = jnp.pad(page_idx, ((0, 0), (0, 0), (0, kp_pad - kp)),
                           constant_values=-1)

    q_nope = q[..., :c_nope].astype(jnp.bfloat16)
    q_rope = q[..., c_nope:].astype(jnp.bfloat16)
    idx = page_idx.astype(jnp.int32)
    sink2d = attn_sink.astype(jnp.float32).reshape(n_h, 1)

    o_nope, o_rope = pl.pallas_call(
        partial(_paged_mqa_kernel, kp_pad=kp_pad, pages=pages, page=page,
                n_win=n_win, s_raw=s_raw, scale=float(c) ** -0.5,
                phi_shift=float(c) ** 0.5, swa_ring=swa_ring),
        grid=(B, T),
        in_specs=[
            pl.BlockSpec((1, 1, kp_pad), lambda b, t: (b, t, 0),
                         memory_space=pltpu.SMEM),
            pl.BlockSpec((1, 1), lambda b, t: (b, t),
                         memory_space=pltpu.SMEM),
            pl.BlockSpec((1, 1), lambda b, t: (b, t),
                         memory_space=pltpu.SMEM),
            pl.BlockSpec((1, 1, n_h, c_nope), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, 1, n_h, rope_dim), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, 1, kp_pad), lambda b, t: (b, t, 0)),
            pl.BlockSpec((n_h, 1), lambda b, t: (0, 0)),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[
            pl.BlockSpec((1, 1, n_h, c_nope), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, 1, n_h, rope_dim), lambda b, t: (b, t, 0, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((B, T, n_h, c_nope), jnp.bfloat16),
            jax.ShapeDtypeStruct((B, T, n_h, rope_dim), jnp.bfloat16),
        ],
        scratch_shapes=[
            pltpu.VMEM((2, pages * page, c_nope), kc.nope.dtype),
            pltpu.VMEM((2, pages * page, rope_dim), jnp.bfloat16),
            pltpu.VMEM((2, pages * page, 1), jnp.float32),
            pltpu.VMEM((n_win, c_nope), swa.nope.dtype),
            pltpu.VMEM((n_win, rope_dim), jnp.bfloat16),
            pltpu.VMEM((n_win, 1), jnp.float32),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
        ),
        interpret=tiles.interpret,
    )(idx, q_pos.astype(jnp.int32), bound.astype(jnp.int32),
      q_nope, q_rope, idx, sink2d,
      kc.nope, kc.rope, kc.scale,
      swa.nope, swa.rope, swa.scale)

    return jnp.concatenate([o_nope, o_rope], axis=-1)
