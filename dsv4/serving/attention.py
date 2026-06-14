"""Fused-gather sparse MQA with sink — the serving attention core.

The HBM problem this kernel exists to solve
-------------------------------------------

``kernel_v1`` materializes the gathered keys as ``K_full[B, T, k+n_win, c]``
in HBM (``jnp.take_along_axis``), then streams them back into the flash
kernel. Per token that is a **write + read of (k+n_win)*c bytes** — at the
Flash config (k=512, n_win=128, c=512, bf16) ~1.3 MB of avoidable traffic
per token *per layer*, and for decode it is the entire latency budget.

Here the compressed-KV cache never leaves HBM in bulk: the kernel receives
it as an unblocked (``memory_space=ANY``) ref and pulls **only the selected
rows** into VMEM with explicit async DMAs, double-buffered so the gather of
chunk j+1 overlaps the MXU work on chunk j (the FlashMLA idea, in TPU
terms). Per token the HBM traffic is the theoretical minimum:
``(k + n_win) * row_bytes`` read, zero intermediate writes.

With the paper's hybrid KV storage (fp8 nope + bf16 rope + f32 row scale,
§2.3.4) ``row_bytes`` is 580 (vs 1024 pure-bf16) — the gather moves ~44%
fewer bytes, on top of the 2x from not materializing ``K_full``.

Layout choices (and the rules forcing them)
-------------------------------------------

- **Split nope/rope arrays end-to-end.** The fp8/bf16 boundary at
  ``c - 64 = 448`` is not lane-aligned, and Mosaic cannot concatenate
  lanes at a non-128 boundary, so the kernel never reconstructs the
  ``c``-wide entry: it runs split dots
  ``logits = q_nope·k_nopeᵀ + q_rope·k_ropeᵀ`` and split PV accumulators,
  and the wrapper concatenates the two output halves in XLA. That concat is
  free *only because* its sole consumer, ``model._grouped_o_proj``, split-
  contracts the o-projection: it slices ``o`` back at the same 448 seam, which
  XLA cancels against this concat (``concat→slice`` = identity), so no
  ``[B,T,n_h,c]`` buffer is materialized. A c-merging reshape in the consumer
  would instead cross the seam and pin the concat to a real buffer — the trap
  the split-contraction avoids.
- **K dequantized to bf16 in VMEM** (``k_q.f32 * row_scale → bf16``).
  This matches the existing bf16 attention baseline's precision (the
  reference itself takes bf16 K) while keeping the MXU on the fast bf16
  path; the softmax state (l, acc) stays fp32 as in kernel_v1.
- **Decoupled phi-softmax, not online-softmax.** Because q and the cached
  k are RMSNormed (``eager.rms_norm``, no learned gain) and RoPE is
  norm-preserving, every logit is bounded: ``|scale·q·kᵀ| ≤ scale·‖q‖‖k‖
  = sqrt(c)`` (Cauchy-Schwarz). So a *constant* shift ``C = sqrt(c)``
  replaces the online running max — ``phi = exp(logit − C) ≤ 1`` never
  overflows, the numerator/denominator are **pure accumulators** (no
  ``acc·alpha`` read-scale-write, no cross-step max/alpha dependency), and
  the shift cancels exactly in the final normalize. Verified equal to the
  softmax oracle to fp32 reassociation noise (~5e-7), ~1e4× under the
  bf16/fp8 quant floor this kernel already carries.
- **Indices live twice**: in SMEM (scalar reads drive DMA addresses) and
  in VMEM (vector compare builds the validity mask). 4*k bytes per token,
  noise.
- **SWA branch is one contiguous DMA** — the window ``[pos-n_win+1, pos]``
  is contiguous in the raw cache, so it costs a single descriptor per
  component, with absolute-position masking instead of zero-padding.

  NOTE: this deliberately *fixes* a reference quirk: ``eager.swa_gather``
  zero-pads the pre-sequence window and counts those zero keys as valid
  (logit 0) softmax entries. This kernel masks them out, which is the
  causally correct math. For tokens with ``pos >= n_win - 1`` the two
  agree exactly; ``serving_attn_ref`` below is the oracle for all tokens.

Grid + work shape
-----------------

Grid is ``(B, T)`` — one program per query token. MQA makes this efficient
for DSv4: the per-token q tile is ``[n_h, c]`` (64x512 Flash, 128x512 Pro),
so each gathered chunk feeds an MXU matmul with M = n_h ≥ 64. Decode is the
same kernel with T=1. (Prefill can alternatively batch tokens per program
by unioning their top-k sets — a future optimization documented in
HARDWARE_NOTES; the XLA-gather v1 path remains available for
throughput-prefill.)

DMA descriptor budget: ``3 * chunk`` descriptors in flight per buffer slot
(nope+rope+scale per row). chunk=128 → 384 outstanding 448/128/4-byte
copies; the row copies are small, which is the price of entry for
arbitrary per-token top-k — if the indexer is ever constrained to
page-aligned selections, swap the per-row copies for per-page ones and
this kernel's gather becomes bandwidth-optimal too.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from .config import ServingTiles
from .quant import KVQuant, dequantize_kv

# Masking sentinel — finite-but-huge negative so exp() flushes to 0 without
# NaN on fully-masked blocks. Re-exported to the sibling attention kernels.
_NEG_INF = -1.0e30


# ---------------------------------------------------------------------------
# Kernel body
# ---------------------------------------------------------------------------


def _gather_chunk_copies(
    idx_smem, base, chunk, slot,
    nope_hbm, rope_hbm, scale_hbm,
    nope_buf, rope_buf, scale_buf, sem, b,
):
    """Build the DMA descriptor list for one gather chunk (one buffer slot).

    Called twice per chunk with identical arguments — once to ``start()``
    and once to ``wait()`` — mirroring the descriptor-rebuild pattern of
    JAX's paged_attention kernel.
    """
    copies = []
    for i in range(chunk):
        idx = jnp.maximum(idx_smem[0, 0, base + i], 0)   # clamp -1 padding
        copies.append(pltpu.make_async_copy(
            nope_hbm.at[b, idx], nope_buf.at[slot, i], sem))
        copies.append(pltpu.make_async_copy(
            rope_hbm.at[b, idx], rope_buf.at[slot, i], sem))
        copies.append(pltpu.make_async_copy(
            scale_hbm.at[b, idx], scale_buf.at[slot, i], sem))
    return copies


def _sparse_mqa_kernel(
    # --- SMEM / scalar-ish inputs ---
    idx_smem_ref,      # [1, 1, k_pad] int32 (SMEM)  gather addresses
    pos_smem_ref,      # [1, 1] int32 (SMEM)         absolute query position
    # --- VMEM blocked inputs ---
    q_nope_ref,        # [1, 1, n_h, c_nope] bf16
    q_rope_ref,        # [1, 1, n_h, r] bf16
    idx_vmem_ref,      # [1, 1, k_pad] int32         validity mask source
    sink_ref,          # [n_h, 1] f32
    # --- HBM (ANY) refs: the caches ---
    kc_nope_hbm,       # [B, S_c, c_nope] e4m3
    kc_rope_hbm,       # [B, S_c, r] bf16
    kc_scale_hbm,      # [B, S_c, 1] f32
    sw_nope_hbm,       # [B, S_r, c_nope] e4m3
    sw_rope_hbm,       # [B, S_r, r] bf16
    sw_scale_hbm,      # [B, S_r, 1] f32
    # --- outputs ---
    o_nope_ref,        # [1, 1, n_h, c_nope] bf16
    o_rope_ref,        # [1, 1, n_h, r] bf16
    # --- scratch ---
    nope_buf,          # [2, chunk, c_nope] e4m3
    rope_buf,          # [2, chunk, r] bf16
    scale_buf,         # [2, chunk, 1] f32
    swn_buf,           # [n_win, c_nope] e4m3
    swr_buf,           # [n_win, r] bf16
    sws_buf,           # [n_win, 1] f32
    gather_sem,
    swa_sem,
    *,
    k_pad: int,
    chunk: int,
    n_win: int,
    s_raw: int,
    scale: float,
    phi_shift: float,
    swa_ring: bool,
):
    b = pl.program_id(0)
    n_chunks = k_pad // chunk
    pos = pos_smem_ref[0, 0]

    # ---- SWA window DMA (single contiguous descriptor per component) ----
    if swa_ring:
        # Dual-write unrolled ring (s_raw == 2*n_win): position p lives at
        # slots p % n_win and p % n_win + n_win, so the window
        # [pos-n_win+1, pos] is ONE contiguous slab starting at
        # (pos+1) % n_win — in position order, exactly like the full-cache
        # path (same single descriptor, same accumulation order).
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

    # ---- kick off gather chunk 0 ----
    mk_copies = partial(
        _gather_chunk_copies,
        idx_smem_ref, nope_hbm=kc_nope_hbm, rope_hbm=kc_rope_hbm,
        scale_hbm=kc_scale_hbm, nope_buf=nope_buf, rope_buf=rope_buf,
        scale_buf=scale_buf, sem=gather_sem, b=b, chunk=chunk,
    )
    for c in mk_copies(0, slot=0):
        c.start()

    q_nope = q_nope_ref[0, 0].astype(jnp.bfloat16)        # [n_h, c_nope]
    q_rope = q_rope_ref[0, 0].astype(jnp.bfloat16)        # [n_h, r]
    n_h = q_nope.shape[0]

    l_i = jnp.zeros((n_h, 1), jnp.float32)
    acc_n = jnp.zeros(o_nope_ref.shape[2:], jnp.float32)  # [n_h, c_nope]
    acc_r = jnp.zeros(o_rope_ref.shape[2:], jnp.float32)  # [n_h, r]

    def phi_update(l_i, acc_n, acc_r, k_n, k_r, valid):
        """One decoupled phi-softmax step over a [rows, ...] key block.

        No running max: the logit is bounded by ``sqrt(c)`` (see module
        docstring), so the constant shift ``C = phi_shift`` stands in for the
        online max. ``phi = exp(logit − C) ≤ 1`` and the accumulators are pure
        adders — the per-step ``acc·alpha`` rescale and the max→alpha→acc
        serial chain are gone; PV dots accumulate straight on the MXU.
        """
        logits = (
            jax.lax.dot_general(q_nope, k_n, (((1,), (1,)), ((), ())),
                                preferred_element_type=jnp.float32)
            + jax.lax.dot_general(q_rope, k_r, (((1,), (1,)), ((), ())),
                                  preferred_element_type=jnp.float32)
        ) * jnp.float32(scale)                            # [n_h, rows]
        phi = jnp.exp(logits - jnp.float32(phi_shift))
        phi = jnp.where(valid, phi, 0.0)                   # masked keys -> 0
        l_new = l_i + jnp.sum(phi, axis=-1, keepdims=True)
        phi_bf = phi.astype(jnp.bfloat16)
        acc_n = acc_n + jax.lax.dot_general(
            phi_bf, k_n, (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32)
        acc_r = acc_r + jax.lax.dot_general(
            phi_bf, k_r, (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32)
        return l_new, acc_n, acc_r

    # ---- top-k gather chunks, double-buffered ----
    for j in range(n_chunks):
        slot = j % 2
        if j + 1 < n_chunks:
            for c in mk_copies((j + 1) * chunk, slot=1 - slot):
                c.start()
        for c in mk_copies(j * chunk, slot=slot):
            c.wait()

        k_n = (nope_buf[slot].astype(jnp.float32)
               * scale_buf[slot]).astype(jnp.bfloat16)    # [chunk, c_nope]
        k_r = rope_buf[slot].astype(jnp.bfloat16)         # [chunk, r]
        idx_vec = idx_vmem_ref[0, 0, j * chunk:(j + 1) * chunk]
        valid = (idx_vec >= 0).reshape(1, chunk)          # [1, chunk]
        l_i, acc_n, acc_r = phi_update(l_i, acc_n, acc_r, k_n, k_r, valid)

    # ---- SWA chunk ----
    for c in swa_copies():
        c.wait()
    k_n = (swn_buf[...].astype(jnp.float32) * sws_buf[...]).astype(jnp.bfloat16)
    k_r = swr_buf[...].astype(jnp.bfloat16)
    if swa_ring:
        # buffer row i holds position pos - n_win + 1 + i by construction;
        # only pre-sequence (negative) positions need masking.
        w_pos = (pos - n_win + 1
                 + jax.lax.broadcasted_iota(jnp.int32, (1, n_win), 1))
        valid = w_pos >= 0
    else:
        w_pos = start + jax.lax.broadcasted_iota(jnp.int32, (1, n_win), 1)
        valid = jnp.logical_and(w_pos <= pos, w_pos > pos - n_win)
    l_i, acc_n, acc_r = phi_update(l_i, acc_n, acc_r, k_n, k_r, valid)

    # ---- finalize with the per-head sink (virtual zero-value logit) ----
    # Same fixed shift C=phi_shift: the sink is a logit, so it joins the
    # denominator as exp(sink - C). The true lse, if ever needed downstream,
    # is C + log(denom) — recoverable exactly.
    sink = sink_ref[...]                                   # [n_h, 1] f32
    denom = l_i + jnp.exp(sink - jnp.float32(phi_shift))
    o_nope_ref[0, 0] = (acc_n / denom).astype(o_nope_ref.dtype)
    o_rope_ref[0, 0] = (acc_r / denom).astype(o_rope_ref.dtype)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


def sparse_mqa_gathered(
    q: jax.Array,            # [B, T, n_h, c] bf16/f32 (RMSNormed + roped)
    kc: KVQuant,             # compressed cache, S_c entries (roped + normed)
    topk_idxs: jax.Array,    # [B, T, k] int32, -1 = invalid
    swa: KVQuant,            # raw SWA cache, S_r entries (roped + normed)
    q_pos: jax.Array,        # [B, T] int32 absolute positions
    attn_sink: jax.Array,    # [n_h] f32
    *,
    n_win: int,
    rope_dim: int = 64,
    swa_ring: bool = False,
    tiles: ServingTiles | None = None,
) -> jax.Array:
    """Returns ``o[B, T, n_h, c]``. See module docstring for semantics.

    ``swa_ring=True`` reads the raw cache as a dual-write unrolled ring of
    ``2 * n_win`` rows (decode); ``False`` reads it as a flat position-
    indexed array (prefill's transient full-length rows)."""
    if tiles is None:
        from .config import tiles_for
        tiles = tiles_for()
    B, T, n_h, c = q.shape
    c_nope = c - rope_dim
    k = topk_idxs.shape[-1]
    chunk = tiles.attn_chunk
    s_raw = swa.nope.shape[1]
    if swa.nope.shape[-1] != c_nope or kc.nope.shape[-1] != c_nope:
        raise ValueError("cache nope width must match q (c - rope_dim)")
    if swa_ring:
        if s_raw != 2 * n_win:
            raise ValueError(f"ring SWA cache must be 2*n_win={2*n_win} rows, "
                             f"got {s_raw}")
    elif s_raw < n_win:
        raise ValueError(f"SWA cache length {s_raw} < n_win={n_win}")

    # Pad k to a chunk multiple with -1 (masked out in-kernel).
    k_pad = ((k + chunk - 1) // chunk) * chunk
    if k_pad != k:
        topk_idxs = jnp.pad(topk_idxs, ((0, 0), (0, 0), (0, k_pad - k)),
                            constant_values=-1)

    q_nope = q[..., :c_nope].astype(jnp.bfloat16)
    q_rope = q[..., c_nope:].astype(jnp.bfloat16)
    idx = topk_idxs.astype(jnp.int32)
    pos = q_pos.astype(jnp.int32)
    sink2d = attn_sink.astype(jnp.float32).reshape(n_h, 1)

    grid = (B, T)
    o_nope, o_rope = pl.pallas_call(
        partial(_sparse_mqa_kernel, k_pad=k_pad, chunk=chunk, n_win=n_win,
                s_raw=s_raw, scale=float(c) ** -0.5, phi_shift=float(c) ** 0.5,
                swa_ring=swa_ring),
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, 1, k_pad), lambda b, t: (b, t, 0),
                         memory_space=pltpu.SMEM),
            pl.BlockSpec((1, 1), lambda b, t: (b, t),
                         memory_space=pltpu.SMEM),
            pl.BlockSpec((1, 1, n_h, c_nope), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, 1, n_h, rope_dim), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, 1, k_pad), lambda b, t: (b, t, 0)),
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
            pltpu.VMEM((2, chunk, c_nope), kc.nope.dtype),
            pltpu.VMEM((2, chunk, rope_dim), jnp.bfloat16),
            pltpu.VMEM((2, chunk, 1), jnp.float32),
            pltpu.VMEM((n_win, c_nope), swa.nope.dtype),
            pltpu.VMEM((n_win, rope_dim), jnp.bfloat16),
            pltpu.VMEM((n_win, 1), jnp.float32),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(
            # Both grid axes are independent: every (b, t) program reads the
            # caches read-only and writes its own disjoint [n_h, c] output tile
            # (decode T=1; prefill T>1). Marking T "parallel" too lets megacore
            # parts split prefill across both cores instead of leaving one idle.
            dimension_semantics=("parallel", "parallel"),
        ),
        interpret=tiles.interpret,
    )(idx, pos, q_nope, q_rope, idx, sink2d,
      kc.nope, kc.rope, kc.scale,
      swa.nope, swa.rope, swa.scale)

    return jnp.concatenate([o_nope, o_rope], axis=-1)


# ---------------------------------------------------------------------------
# Eager oracle (dense, fp32 — what the kernel is graded against)
# ---------------------------------------------------------------------------


def serving_attn_ref(
    q: jax.Array,            # [B, T, n_h, c]
    kc: KVQuant,
    topk_idxs: jax.Array,    # [B, T, k]
    swa: KVQuant,
    q_pos: jax.Array,        # [B, T]
    attn_sink: jax.Array,    # [n_h]
    *,
    n_win: int,
    rope_dim: int = 64,
) -> jax.Array:
    """Dense reference with the same dequantization + causally-correct SWA
    masking (no zero-pad pseudo-keys) the kernel implements."""
    B, T, n_h, c = q.shape
    scale = float(c) ** -0.5
    K_comp = dequantize_kv(kc)                              # [B, S_c, c] f32
    K_raw = dequantize_kv(swa)                              # [B, S_r, c] f32
    qf = q.astype(jnp.float32)

    safe = jnp.maximum(topk_idxs, 0)
    K_sel = jax.vmap(lambda kcb, idxb: kcb[idxb])(K_comp, safe)  # [B,T,k,c]
    sel_valid = topk_idxs >= 0                               # [B, T, k]

    # SWA window rows per (b, t): absolute positions pos-n_win+1 .. pos.
    w_off = jnp.arange(n_win) - (n_win - 1)                  # [-n_win+1 .. 0]
    w_pos = q_pos[..., None] + w_off[None, None, :]          # [B, T, n_win]
    w_safe = jnp.clip(w_pos, 0, K_raw.shape[1] - 1)
    K_win = jax.vmap(lambda kb, idxb: kb[idxb])(K_raw, w_safe)   # [B,T,n_win,c]
    win_valid = w_pos >= 0

    K_all = jnp.concatenate([K_sel, K_win], axis=2)
    valid = jnp.concatenate([sel_valid, win_valid], axis=2)  # [B, T, S]

    logits = jnp.einsum("bthc,btsc->bths", qf, K_all) * scale
    logits = jnp.where(valid[:, :, None, :], logits, -jnp.inf)
    m = jnp.max(logits, axis=-1, keepdims=True)
    sink = attn_sink.astype(jnp.float32)[None, None, :, None]
    m = jnp.maximum(m, sink)
    p = jnp.exp(logits - m)
    p = jnp.where(valid[:, :, None, :], p, 0.0)
    denom = p.sum(axis=-1, keepdims=True) + jnp.exp(sink - m)
    return jnp.einsum("bths,btsc->bthc", p / denom, K_all)
