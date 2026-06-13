"""Valid-prefix lightning-indexer scan — the decode-side scan kernel.

The HBM problem this kernel exists to solve
-------------------------------------------

The lightning indexer is the *one* per-token operation that touches the
whole context: every decode step scores all completed compressed blocks
(``score[s] = Σ_h w_h · ReLU(q_h · k_I[s])``) to pick the top-k. The
roofline (HARDWARE_NOTES §11.1) says this scan is 80–93% of all
long-context bytes — which is why the ki cache went fp8 (§13.1) and
paged (§13.3).

But there is a third factor the XLA path cannot fix: **static shapes**.
``_indexer_scores_step`` is an einsum over the *allocated* cache
``[B, S_max/m, c_I]``; XLA must read every row of it on every step, even
when the sequence is at position 64K inside a 1M-row allocation. Bytes
scale with what you *reserved*, not what you have *written* — a 16x
overscan in that example, on the dominant term (§12 item 6).

Pallas can do what XLA cannot: make the byte count data-dependent. The
cache stays in HBM as an unblocked (``memory_space=ANY``) ref, the grid
walks it in fixed-size chunks, and each chunk's DMA is issued **under
``pl.when(chunk_start < n_valid)``** — a chunk past the valid prefix
never leaves HBM; its program just fills the score block with -inf (the
exact sentinel the eager mask produces, so ``eager.topk_indices``'
isfinite contract is untouched). Actual traffic becomes
``ceil(n_valid/chunk) * chunk`` rows: bytes track *position*, matching
the cost the roofline tables already model (``dyn()`` charges
``S//m`` entries — this kernel is what makes hardware agree with that).

Numerics
--------

Bit-for-bit the same math as the eager reference, in the same places:
the tiny query projections (``h @ W_DQ @ W_IUQ``, ``h @ W_w``) stay in
XLA in the model dtype; per-row dequant is ``fp8 → f32 · row_scale``
(== ``quant.dequantize_rows``); the q·k dot, ReLU, and head-weighted sum
all accumulate in f32. The only freedom left is dot-product reduction
order (MXU tiling vs einsum), ~1 ulp f32 — far below the fp8 payload
rounding the recall test already budgets for.

Work shape
----------

Grid is ``(B, S_blk // chunk)`` — one program per (sequence, chunk);
chunk = 128 rows (or the full cache when S_blk < 128, the small-test
case). Per program: 2 DMAs (payload slab + scale slab, contiguous — the
descriptor-efficient regime, unlike the gather's per-row copies), one
``[n_I_h, c_I] x [c_I, chunk]`` MXU dot, one masked store. There is no
cross-chunk carry — scores are per-row independent — so chunks need no
double-buffering loop; the (sequential) grid lets Mosaic overlap chunk
j+1's DMA with chunk j's epilogue.

Used by ``model._attn_decode`` for both the row-exact scan (``cache.ki``,
n_valid = pos//m) and the paged summary scan (``cache.kis``, n_valid =
(pos//m)//P). The eager ``model._indexer_scores_step`` remains the test
oracle.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from .config import ServingTiles
from .quant import RowQuant


# ---------------------------------------------------------------------------
# Kernel body
# ---------------------------------------------------------------------------


def _ki_scan_kernel(
    # --- SMEM scalar inputs ---
    nv_smem_ref,       # [1, 1] int32 (SMEM)   valid-prefix row count
    # --- VMEM blocked inputs ---
    qi_ref,            # [1, n_I_h, c_I] f32   indexer queries (one token)
    wi_ref,            # [1, n_I_h] f32        per-head mixture weights
    # --- HBM (ANY) refs ---
    kiq_hbm,           # [B, S_blk, c_I] e4m3  ki payload cache
    kis_hbm,           # [B, S_blk, 1] f32     ki row scales
    # --- outputs ---
    out_ref,           # [1, chunk] f32        scores for this chunk
    # --- scratch ---
    q_buf,             # [chunk, c_I] e4m3
    s_buf,             # [chunk, 1] f32
    sem,               # DMA semaphore
    *,
    chunk: int,
):
    b = pl.program_id(0)
    ci = pl.program_id(1)
    n_valid = nv_smem_ref[0, 0]
    base = ci * chunk
    rows = base + jax.lax.broadcasted_iota(jnp.int32, (1, chunk), 1)

    def chunk_copies():
        # Two contiguous slab descriptors; rebuilt identically for start()
        # and wait() (the descriptor-rebuild pattern, attention.py).
        return [
            pltpu.make_async_copy(
                kiq_hbm.at[b, pl.ds(base, chunk)], q_buf, sem),
            pltpu.make_async_copy(
                kis_hbm.at[b, pl.ds(base, chunk)], s_buf, sem),
        ]

    @pl.when(base < n_valid)
    def _scan():
        # The whole point: this DMA does not exist for chunks past the
        # valid prefix — HBM bytes track position, not allocation.
        for c in chunk_copies():
            c.start()
        for c in chunk_copies():
            c.wait()
        k = q_buf[...].astype(jnp.float32) * s_buf[...]   # dequantize_rows
        qk = jax.lax.dot_general(                          # [n_I_h, chunk]
            qi_ref[...].reshape(-1, k.shape[-1]), k,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32)
        sc = (wi_ref[...].reshape(-1, 1)
              * jax.nn.relu(qk)).sum(axis=0, keepdims=True)
        # Partial chunk straddling the prefix edge: mask rows past it.
        out_ref[...] = jnp.where(rows < n_valid, sc, -jnp.inf)

    @pl.when(base >= n_valid)
    def _skip():
        out_ref[...] = jnp.full((1, chunk), -jnp.inf, jnp.float32)


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


def indexer_scores_decode(h_t, ki: RowQuant, p, cfg_csa, n_valid, *,
                          tiles: ServingTiles):
    """Lightning-indexer scores for one decode token, reading only the
    ``n_valid``-row prefix of the fp8 ki cache.

    Drop-in for ``model._indexer_scores_step(h_t, dequantize_rows(ki),
    ...)``: same [B, 1, S_blk] f32 result, -inf at rows >= n_valid, but
    HBM traffic ∝ n_valid instead of ∝ S_blk (the allocation).

    ``ki`` may be the row cache (n_valid = pos//m) or the page-summary
    cache (n_valid = (pos//m)//P) — the kernel is agnostic.
    """
    B = h_t.shape[0]
    S_blk, c_I = ki.q.shape[1], ki.q.shape[2]
    n_I_h = cfg_csa.n_I_h
    chunk = min(S_blk, 128)
    if S_blk % chunk:
        raise ValueError(
            f"ki cache rows ({S_blk}) must be a multiple of the scan chunk "
            f"({chunk}); size s_max so s_max//m (and //pages) is 128-aligned "
            "or smaller than 128.")

    # Query projections: tiny GEMMs, identical dtype path to the eager
    # reference (model dtype matmul, then f32) — stay in XLA.
    cQ = h_t @ p.W_DQ                                     # [B, 1, d_c]
    qi = (cQ @ p.W_IUQ).reshape(B, n_I_h, c_I).astype(jnp.float32)
    wi = (h_t @ p.W_w).reshape(B, n_I_h).astype(jnp.float32)
    nv = jnp.full((B, 1), n_valid, jnp.int32)

    scores = pl.pallas_call(
        partial(_ki_scan_kernel, chunk=chunk),
        grid=(B, S_blk // chunk),
        in_specs=[
            pl.BlockSpec((1, 1), lambda b, ci: (b, 0),
                         memory_space=pltpu.SMEM),
            pl.BlockSpec((1, n_I_h, c_I), lambda b, ci: (b, 0, 0)),
            pl.BlockSpec((1, n_I_h), lambda b, ci: (b, 0)),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec((1, chunk), lambda b, ci: (b, ci)),
        out_shape=jax.ShapeDtypeStruct((B, S_blk), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((chunk, c_I), ki.q.dtype),
            pltpu.VMEM((chunk, 1), jnp.float32),
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
        ),
        interpret=tiles.interpret,
    )(nv, qi, wi, ki.q, ki.scale)
    return scores[:, None, :]


# ---------------------------------------------------------------------------
# Envelope upper-bound scan (exact coarse-to-fine, HARDWARE_NOTES §13.7)
# ---------------------------------------------------------------------------
#
# Same chunked valid-prefix walk, different math: instead of scoring mean
# summaries, score *sound per-page upper bounds*. With the per-page
# coordinatewise envelopes  k_hi >= k_s >= k_lo  (all rows s in the page),
# any query splits as q = q+ + q-  (q+ = ReLU(q) >= 0, q- <= 0), giving
#
#   q . k_s  <=  q+ . k_hi + q- . k_lo   =: ub
#   q . k_s  >=  q+ . k_lo + q- . k_hi   =: lb
#
# and since w splits the same way and ReLU is monotone,
#
#   I(s) = SUM_h w_h ReLU(q_h . k_s)
#        <= SUM_h [ w+_h ReLU(ub_h) - (-w-_h) ReLU(lb_h) ]  =: UB(page).
#
# The envelopes are fp8 rows stored with *directed* rounding
# (quant.quantize_rows_bound): their plain f32 dequant — exactly the
# product this kernel computes — already dominates the pre-quantization
# value, so UB stays sound end-to-end with no inflation pad. Pages whose
# UB survives the coarse top-R are rescanned at row resolution
# (model.py) — selection is then provably the row-exact f32 top-k
# whenever the candidate set covers it (the certificate the tests
# assert).


def _ub_scan_kernel(
    nv_smem_ref,       # [1, 1] int32 (SMEM)   valid-prefix page count
    qp_ref,            # [1, n_I_h, c_I] f32   ReLU(q)
    qn_ref,            # [1, n_I_h, c_I] f32   q - ReLU(q)  (<= 0)
    wp_ref,            # [1, n_I_h] f32        ReLU(w)
    wn_ref,            # [1, n_I_h] f32        ReLU(-w)
    hi_q_hbm,          # [B, S_pg, c_I] e4m3   max-envelope payload
    hi_s_hbm,          # [B, S_pg, 1] f32
    lo_q_hbm,          # [B, S_pg, c_I] e4m3   min-envelope payload
    lo_s_hbm,          # [B, S_pg, 1] f32
    out_ref,           # [1, chunk] f32        page UBs for this chunk
    hi_qb, hi_sb, lo_qb, lo_sb,                # scratch
    sem,
    *,
    chunk: int,
):
    b = pl.program_id(0)
    ci = pl.program_id(1)
    n_valid = nv_smem_ref[0, 0]
    base = ci * chunk
    rows = base + jax.lax.broadcasted_iota(jnp.int32, (1, chunk), 1)

    def chunk_copies():
        return [
            pltpu.make_async_copy(
                hi_q_hbm.at[b, pl.ds(base, chunk)], hi_qb, sem),
            pltpu.make_async_copy(
                hi_s_hbm.at[b, pl.ds(base, chunk)], hi_sb, sem),
            pltpu.make_async_copy(
                lo_q_hbm.at[b, pl.ds(base, chunk)], lo_qb, sem),
            pltpu.make_async_copy(
                lo_s_hbm.at[b, pl.ds(base, chunk)], lo_sb, sem),
        ]

    @pl.when(base < n_valid)
    def _scan():
        for c in chunk_copies():
            c.start()
        for c in chunk_copies():
            c.wait()
        # Plain dequant is already a sound bound: the envelopes were
        # stored with directed rounding (quant.quantize_rows_bound).
        k_hi = hi_qb[...].astype(jnp.float32) * hi_sb[...]
        k_lo = lo_qb[...].astype(jnp.float32) * lo_sb[...]

        def dotk(q2, k2):
            return jax.lax.dot_general(
                q2.reshape(-1, k2.shape[-1]), k2, (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32)   # [n_I_h, chunk]

        ub = dotk(qp_ref[...], k_hi) + dotk(qn_ref[...], k_lo)
        lb = dotk(qp_ref[...], k_lo) + dotk(qn_ref[...], k_hi)
        sc = (wp_ref[...].reshape(-1, 1) * jax.nn.relu(ub)
              - wn_ref[...].reshape(-1, 1) * jax.nn.relu(lb)
              ).sum(axis=0, keepdims=True)
        out_ref[...] = jnp.where(rows < n_valid, sc, -jnp.inf)

    @pl.when(base >= n_valid)
    def _skip():
        out_ref[...] = jnp.full((1, chunk), -jnp.inf, jnp.float32)


def indexer_ub_scores_decode(h_t, kis_hi: RowQuant, kis_lo: RowQuant,
                             p, cfg_csa, n_valid, *, tiles: ServingTiles):
    """Sound per-page score upper bounds for one decode token.

    Returns [B, 1, S_pg] f32 with -inf at pages >= n_valid. Reads only the
    valid prefix of BOTH envelope caches (same chunked pl.when walk as
    ``indexer_scores_decode``). UB(page) >= I(row) for every row in the
    page, including all fp8 storage rounding.
    """
    B = h_t.shape[0]
    S_pg, c_I = kis_hi.q.shape[1], kis_hi.q.shape[2]
    n_I_h = cfg_csa.n_I_h
    chunk = min(S_pg, 128)
    if S_pg % chunk:
        raise ValueError(
            f"envelope cache rows ({S_pg}) must be a multiple of the scan "
            f"chunk ({chunk}); size s_max so s_max//m//pages is 128-aligned "
            "or smaller than 128.")

    cQ = h_t @ p.W_DQ
    qi = (cQ @ p.W_IUQ).reshape(B, n_I_h, c_I).astype(jnp.float32)
    wi = (h_t @ p.W_w).reshape(B, n_I_h).astype(jnp.float32)
    qp = jax.nn.relu(qi)
    qn = qi - qp
    wp = jax.nn.relu(wi)
    wn = jax.nn.relu(-wi)
    nv = jnp.full((B, 1), n_valid, jnp.int32)

    scores = pl.pallas_call(
        partial(_ub_scan_kernel, chunk=chunk),
        grid=(B, S_pg // chunk),
        in_specs=[
            pl.BlockSpec((1, 1), lambda b, ci: (b, 0),
                         memory_space=pltpu.SMEM),
            pl.BlockSpec((1, n_I_h, c_I), lambda b, ci: (b, 0, 0)),
            pl.BlockSpec((1, n_I_h, c_I), lambda b, ci: (b, 0, 0)),
            pl.BlockSpec((1, n_I_h), lambda b, ci: (b, 0)),
            pl.BlockSpec((1, n_I_h), lambda b, ci: (b, 0)),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec((1, chunk), lambda b, ci: (b, ci)),
        out_shape=jax.ShapeDtypeStruct((B, S_pg), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((chunk, c_I), kis_hi.q.dtype),
            pltpu.VMEM((chunk, 1), jnp.float32),
            pltpu.VMEM((chunk, c_I), kis_lo.q.dtype),
            pltpu.VMEM((chunk, 1), jnp.float32),
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
        ),
        interpret=tiles.interpret,
    )(nv, qp, qn, wp, wn, kis_hi.q, kis_hi.scale, kis_lo.q, kis_lo.scale)
    return scores[:, None, :]
