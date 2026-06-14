"""Q-block batched sparse MQA — the prefill gather amortizer (§13.8).

The per-token gather kernel (`attention.sparse_mqa_gathered`, grid (B, T))
re-issues, for *every* query token, a fresh DMA gather of its top-k rows
plus its n_win SWA window. Consecutive prefill tokens overlap massively:
their SWA windows slide by one row, and — because attention is locally
smooth — their top-k sets are nearly identical. So the per-token kernel
pays for the same HBM rows again and again.

This kernel batches a block of BQ consecutive query tokens into one
program (the splash-attention idea):

  - **top-k union, gathered once.** The block's BQ·k selected indices are
    deduplicated into a compacted union of D distinct rows (D ≤ BQ·k),
    gathered a single time into shared VMEM. Per token, a membership mask
    restricts attention to *exactly* that token's own top-k — so the math
    is identical to the per-token kernel (same rows, same masking), only
    the DMA is shared. With the §13.5 trick the chunk DMAs run under
    ``pl.when(chunk_start < D)``, so the gather moves D rows, not the
    static BQ·k: duplicate elimination becomes a real bandwidth cut
    (D ≈ k–2k in practice vs BQ·k issued by the per-token path).
  - **SWA span, gathered once.** The block's tokens span positions
    ``[p0, p0+BQ)``; their windows together cover the contiguous slab
    ``[p0−n_win+1, p0+BQ−1]`` — ``n_win + BQ − 1`` rows read once instead
    of BQ overlapping n_win-row reads. Per-token positional masking
    recovers each token's own window.

Exactness (no quality trade): K_union = round_up(BQ·k) ≥ the true union,
so no selected row is ever dropped; each token attends precisely its
top-k. Numerically identical to the per-token kernel up to flash
reassociation order (the union is processed sorted, not in topk order) —
within the decode≡prefill gold tolerance. **Prefill only** (positions
contiguous from 0, flat SWA cache); decode keeps the per-token kernel.

Scope note: the membership preamble below is O(BQ·k·K_union) per block —
fine at the dev/test scale, and a searchsorted/bitmap reduction is the
production refinement (the Pallas kernel itself is production-shaped).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from .attention import _gather_chunk_copies
from .config import ServingTiles
from .quant import KVQuant


def _blocked_mqa_kernel(
    union_smem_ref,    # [1, 1, K_union] int32 (SMEM) gather addresses
    d_smem_ref,        # [1, 1] int32 (SMEM)          distinct union count
    q_nope_ref,        # [1, 1, BQ, n_h, c_nope] bf16
    q_rope_ref,        # [1, 1, BQ, n_h, r] bf16
    memb_ref,          # [1, 1, BQ, K_union] int32    per-token membership
    sink_ref,          # [n_h, 1] f32
    kc_nope_hbm, kc_rope_hbm, kc_scale_hbm,
    sw_nope_hbm, sw_rope_hbm, sw_scale_hbm,
    o_nope_ref,        # [1, 1, BQ, n_h, c_nope] bf16
    o_rope_ref,        # [1, 1, BQ, n_h, r] bf16
    nope_buf, rope_buf, scale_buf,   # [2, chunk, ...]
    swn_buf, swr_buf, sws_buf,       # [span, ...]
    accn_buf, accr_buf, l_buf,       # [bq, n_h, ...] f32 per-token accumulators
    gather_sem, swa_sem,
    *,
    k_pad: int,
    chunk: int,
    n_win: int,
    span: int,
    s_raw: int,
    scale: float,
    phi_shift: float,
    bq: int,
):
    b = pl.program_id(0)
    blk = pl.program_id(1)
    D = d_smem_ref[0, 0]
    p0 = blk * bq
    n_chunks = k_pad // chunk

    # Finite-safety: skipped chunks (start >= D) leave their buffer slot
    # holding either a previously-gathered finite chunk or this zero — so
    # masked rows feed 0 into the PV dot, never a stale NaN/Inf (0*x=0).
    nope_buf[...] = jnp.zeros(nope_buf.shape, nope_buf.dtype)
    rope_buf[...] = jnp.zeros(rope_buf.shape, rope_buf.dtype)
    scale_buf[...] = jnp.zeros(scale_buf.shape, scale_buf.dtype)

    # ---- SWA span DMA (one contiguous slab for the whole block) ----
    span_start = jnp.clip(p0 - n_win + 1, 0, s_raw - span)

    def swa_copies():
        return [
            pltpu.make_async_copy(
                sw_nope_hbm.at[b, pl.ds(span_start, span)], swn_buf, swa_sem),
            pltpu.make_async_copy(
                sw_rope_hbm.at[b, pl.ds(span_start, span)], swr_buf, swa_sem),
            pltpu.make_async_copy(
                sw_scale_hbm.at[b, pl.ds(span_start, span)], sws_buf, swa_sem),
        ]
    for c in swa_copies():
        c.start()

    mk = partial(
        _gather_chunk_copies,
        union_smem_ref, nope_hbm=kc_nope_hbm, rope_hbm=kc_rope_hbm,
        scale_hbm=kc_scale_hbm, nope_buf=nope_buf, rope_buf=rope_buf,
        scale_buf=scale_buf, sem=gather_sem, b=b, chunk=chunk,
    )

    @pl.when(0 < D)
    def _start0():
        for c in mk(0, slot=0):
            c.start()

    q_nope_all = q_nope_ref[0, 0]                         # [BQ, n_h, c_nope]
    q_rope_all = q_rope_ref[0, 0]                         # [BQ, n_h, r]
    n_h, cn = q_nope_all.shape[1], q_nope_all.shape[2]
    r = q_rope_all.shape[2]
    M = bq * n_h
    # All BQ tokens of the block attend the SAME gathered K (shared RHS), so
    # stack their per-head queries on the M axis and issue ONE [M, c]·[c, chunk]
    # matmul per chunk instead of BQ at M=n_h. On Flash (n_h=64) this lifts M
    # 64 -> 512, saturating the 128-row MXU; bit-exact because each output row
    # is an independent dot over c (stacking M rows changes no single row).
    q_nope_flat = q_nope_all.reshape(M, cn)
    q_rope_flat = q_rope_all.reshape(M, r)

    def phi_update(st, q_nope, q_rope, k_n, k_r, valid):
        # Decoupled phi-softmax (constant shift C=phi_shift, see attention.py):
        # pure accumulators, no running max / acc*alpha rescale.
        l_i, acc_n, acc_r = st
        logits = (
            jax.lax.dot_general(q_nope, k_n, (((1,), (1,)), ((), ())),
                                preferred_element_type=jnp.float32)
            + jax.lax.dot_general(q_rope, k_r, (((1,), (1,)), ((), ())),
                                  preferred_element_type=jnp.float32)
        ) * jnp.float32(scale)
        phi = jnp.where(valid, jnp.exp(logits - jnp.float32(phi_shift)), 0.0)
        l_new = l_i + jnp.sum(phi, axis=-1, keepdims=True)
        phi_bf = phi.astype(jnp.bfloat16)
        acc_n = acc_n + jax.lax.dot_general(
            phi_bf, k_n, (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32)
        acc_r = acc_r + jax.lax.dot_general(
            phi_bf, k_r, (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32)
        return l_new, acc_n, acc_r

    # Per-token accumulators live in scratch (not a python carry) so the
    # union-chunk MXU can be skipped wholesale under pl.when when the chunk is
    # beyond the distinct count D. Bit-identical to running phi_update on the
    # all-masked chunk (every membership entry past D is 0 -> phi=0 -> +0), but
    # the QK/PV dots never execute. Zero-init is the additive identity.
    accn_buf[...] = jnp.zeros(accn_buf.shape, jnp.float32)
    accr_buf[...] = jnp.zeros(accr_buf.shape, jnp.float32)
    l_buf[...] = jnp.zeros(l_buf.shape, jnp.float32)

    # ---- union chunks, double-buffered; DMA *and* MXU gated by the count ----
    for j in range(n_chunks):
        slot = j % 2
        base = j * chunk
        if j + 1 < n_chunks:
            @pl.when((j + 1) * chunk < D)
            def _prefetch(s=1 - slot, nb=(j + 1) * chunk):
                for c in mk(nb, slot=s):
                    c.start()

        # base < D => the chunk holds >=1 distinct row: wait, dequant, run the
        # bq QK/PV dots. base >= D => the whole chunk is past the union, so skip
        # it entirely (its DMA was never started; its contribution would be 0).
        @pl.when(base < D)
        def _chunk(s=slot, bb=base):
            for c in mk(bb, slot=s):
                c.wait()
            k_n = (nope_buf[s].astype(jnp.float32)
                   * scale_buf[s]).astype(jnp.bfloat16)
            k_r = rope_buf[s].astype(jnp.bfloat16)
            # per-token membership [bq, chunk] -> [M, chunk] (shared over heads)
            memb = (memb_ref[0, 0, :, bb:bb + chunk] != 0)
            valid = jnp.broadcast_to(memb[:, None, :],
                                     (bq, n_h, chunk)).reshape(M, chunk)
            l_new, an, ar = phi_update(
                (l_buf[...].reshape(M, 1), accn_buf[...].reshape(M, cn),
                 accr_buf[...].reshape(M, r)),
                q_nope_flat, q_rope_flat, k_n, k_r, valid)
            l_buf[...] = l_new.reshape(bq, n_h, 1)
            accn_buf[...] = an.reshape(bq, n_h, cn)
            accr_buf[...] = ar.reshape(bq, n_h, r)

    # ---- SWA span chunk (shared slab, per-token positional mask) ----
    for c in swa_copies():
        c.wait()
    k_n = (swn_buf[...].astype(jnp.float32) * sws_buf[...]).astype(jnp.bfloat16)
    k_r = swr_buf[...].astype(jnp.bfloat16)
    span_iota = jax.lax.broadcasted_iota(jnp.int32, (1, span), 1)   # [1, span]
    p_t = p0 + jax.lax.broadcasted_iota(jnp.int32, (bq, 1), 0)      # [bq, 1] token pos
    w_pos = span_start + span_iota                                  # [1, span]
    memb = jnp.logical_and(jnp.logical_and(w_pos <= p_t, w_pos > p_t - n_win),
                           w_pos >= 0)                              # [bq, span]
    valid = jnp.broadcast_to(memb[:, None, :], (bq, n_h, span)).reshape(M, span)
    l_new, an, ar = phi_update(
        (l_buf[...].reshape(M, 1), accn_buf[...].reshape(M, cn),
         accr_buf[...].reshape(M, r)),
        q_nope_flat, q_rope_flat, k_n, k_r, valid)
    l_buf[...] = l_new.reshape(bq, n_h, 1)
    accn_buf[...] = an.reshape(bq, n_h, cn)
    accr_buf[...] = ar.reshape(bq, n_h, r)

    # ---- finalize each token with the per-head sink ----
    # exp(sink - C) is grid-invariant; compute it once, not bq times.
    sink_term = jnp.exp(sink_ref[...] - jnp.float32(phi_shift))   # [n_h, 1]
    for t in range(bq):
        denom = l_buf[t] + sink_term
        o_nope_ref[0, 0, t] = (accn_buf[t] / denom).astype(o_nope_ref.dtype)
        o_rope_ref[0, 0, t] = (accr_buf[t] / denom).astype(o_rope_ref.dtype)


def _build_union(topk_idxs, bq, k_union):
    """(union_idx [B,NB,K_union], D [B,NB], memb [B,NB,BQ,K_union]).

    Dedup+compact the block's BQ·k selected indices into a front-packed
    union (−1 padded), its distinct count, and the per-token membership
    mask. Pure XLA, runs in the prefill preamble."""
    B, T_pad, k = topk_idxs.shape
    nb = T_pad // bq
    tb = topk_idxs.reshape(B, nb, bq, k)
    flat = tb.reshape(B, nb, bq * k)
    sflat = jnp.sort(flat, axis=-1)                        # −1s sort to front
    prev = jnp.concatenate(
        [jnp.full((B, nb, 1), -2, sflat.dtype), sflat[..., :-1]], axis=-1)
    uniq = jnp.where((sflat != prev) & (sflat >= 0), sflat, -1)
    order = jnp.argsort(uniq < 0, axis=-1, stable=True)    # valid (False) first
    comp = jnp.take_along_axis(uniq, order, axis=-1)
    comp = jnp.pad(comp, ((0, 0), (0, 0), (0, k_union - bq * k)),
                   constant_values=-1).astype(jnp.int32)
    d = (comp >= 0).sum(-1).astype(jnp.int32)              # [B, nb]
    memb = (tb[:, :, :, None, :] == comp[:, :, None, :, None]).any(-1)
    memb = (memb & (comp[:, :, None, :] >= 0)).astype(jnp.int32)
    return comp, d, memb


def sparse_mqa_gathered_blocked(
    q: jax.Array,            # [B, T, n_h, c]
    kc: KVQuant,
    topk_idxs: jax.Array,    # [B, T, k] int32, -1 = invalid
    swa: KVQuant,            # flat prefill SWA rows (position-indexed)
    attn_sink: jax.Array,    # [n_h] f32
    *,
    n_win: int,
    rope_dim: int = 64,
    bq: int = 8,
    tiles: ServingTiles | None = None,
) -> jax.Array:
    """Prefill-only q-block batched gather. Returns ``o[B, T, n_h, c]``,
    numerically equal (up to flash reassociation) to
    ``attention.sparse_mqa_gathered`` but amortizing the gather over BQ
    tokens. Assumes contiguous positions 0..T-1 (the prefill contract)."""
    if tiles is None:
        from .config import tiles_for
        tiles = tiles_for()
    B, T, n_h, c = q.shape
    c_nope = c - rope_dim
    k = topk_idxs.shape[-1]
    chunk = tiles.attn_chunk
    if swa.nope.shape[-1] != c_nope or kc.nope.shape[-1] != c_nope:
        raise ValueError("cache nope width must match q (c - rope_dim)")

    # Pad T to a BQ multiple (extra query rows masked, sliced off at the end).
    t_pad = ((T + bq - 1) // bq) * bq
    nb = t_pad // bq
    if t_pad != T:
        q = jnp.pad(q, ((0, 0), (0, t_pad - T), (0, 0), (0, 0)))
        topk_idxs = jnp.pad(topk_idxs, ((0, 0), (0, t_pad - T), (0, 0)),
                            constant_values=-1)

    k_union = ((bq * k + chunk - 1) // chunk) * chunk
    union_idx, d, memb = _build_union(topk_idxs.astype(jnp.int32), bq, k_union)

    # SWA slab covering a block: n_win + BQ - 1 rows, rounded to a sublane
    # multiple; source padded so the slab never under-runs.
    span = ((n_win + bq - 1 + 7) // 8) * 8
    s_raw = swa.nope.shape[1]
    if s_raw < span:
        pad = span - s_raw
        swa = jax.tree.map(
            lambda a: jnp.pad(a, ((0, 0), (0, pad), (0, 0))), swa)
        s_raw = span

    q_nope = q[..., :c_nope].astype(jnp.bfloat16).reshape(B, nb, bq, n_h, c_nope)
    q_rope = q[..., c_nope:].astype(jnp.bfloat16).reshape(B, nb, bq, n_h, rope_dim)
    union_b = union_idx.reshape(B, nb, k_union)
    sink2d = attn_sink.astype(jnp.float32).reshape(n_h, 1)

    o_nope, o_rope = pl.pallas_call(
        partial(_blocked_mqa_kernel, k_pad=k_union, chunk=chunk, n_win=n_win,
                span=span, s_raw=s_raw, scale=float(c) ** -0.5,
                phi_shift=float(c) ** 0.5, bq=bq),
        grid=(B, nb),
        in_specs=[
            pl.BlockSpec((1, 1, k_union), lambda b, t: (b, t, 0),
                         memory_space=pltpu.SMEM),
            pl.BlockSpec((1, 1), lambda b, t: (b, t),
                         memory_space=pltpu.SMEM),
            pl.BlockSpec((1, 1, bq, n_h, c_nope), lambda b, t: (b, t, 0, 0, 0)),
            pl.BlockSpec((1, 1, bq, n_h, rope_dim), lambda b, t: (b, t, 0, 0, 0)),
            pl.BlockSpec((1, 1, bq, k_union), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((n_h, 1), lambda b, t: (0, 0)),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[
            pl.BlockSpec((1, 1, bq, n_h, c_nope), lambda b, t: (b, t, 0, 0, 0)),
            pl.BlockSpec((1, 1, bq, n_h, rope_dim), lambda b, t: (b, t, 0, 0, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((B, nb, bq, n_h, c_nope), jnp.bfloat16),
            jax.ShapeDtypeStruct((B, nb, bq, n_h, rope_dim), jnp.bfloat16),
        ],
        scratch_shapes=[
            pltpu.VMEM((2, chunk, c_nope), kc.nope.dtype),
            pltpu.VMEM((2, chunk, rope_dim), jnp.bfloat16),
            pltpu.VMEM((2, chunk, 1), jnp.float32),
            pltpu.VMEM((span, c_nope), swa.nope.dtype),
            pltpu.VMEM((span, rope_dim), jnp.bfloat16),
            pltpu.VMEM((span, 1), jnp.float32),
            pltpu.VMEM((bq, n_h, c_nope), jnp.float32),    # accn_buf
            pltpu.VMEM((bq, n_h, rope_dim), jnp.float32),  # accr_buf
            pltpu.VMEM((bq, n_h, 1), jnp.float32),         # l_buf
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
        ),
        interpret=tiles.interpret,
    )(union_b, d, q_nope, q_rope, memb, sink2d,
      kc.nope, kc.rope, kc.scale,
      swa.nope, swa.rope, swa.scale)

    o = jnp.concatenate([o_nope, o_rope], axis=-1)         # [B, nb, bq, n_h, c]
    return o.reshape(B, t_pad, n_h, c)[:, :T]
