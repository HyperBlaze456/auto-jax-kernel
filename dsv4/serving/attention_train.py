"""Training-path gather attention: forward (o, lse) + memory-efficient backward.

Relationship to ``attention.py``: the serving kernel reads the *quantized*
hybrid KV cache; training keeps K in bf16 (quantization of the KV cache is
a serving-time storage decision — paper QAT touches MoE weights and the
indexer, not the attention KV during training). With a single bf16 K array
the nope/rope split disappears and both kernels get materially simpler.

Forward
-------

Same gather discipline as serving: K_comp/K_swa stay in HBM (``ANY``
refs); per-token programs pull only the selected rows with double-buffered
row DMAs; online softmax with the per-head sink in fp32. Additionally
emits ``lse[B, T, n_h, 1]`` (sink-inclusive log-sum-exp) — the only
attention residual. Activation memory per token: ``n_h·c`` (o, saved by
JAX anyway) + ``n_h`` floats of lse. K_full is never materialized,
forward or backward.

Backward (kernel_refs §D3 recompute, paper §3.3 determinism)
------------------------------------------------------------

Per token, the bwd kernel **re-gathers** the same K rows (second pass over
the same minimal byte set), recomputes ``p = exp(scale·qk − lse)``, and
applies the softmax-backprop identity ``dlogits = p ∘ (dp − D)`` with
``D = Σ_c dout·o`` precomputed in XLA. It writes:

  - ``dq``               — accumulated across chunks in-register, one write;
  - ``dk_contrib[B,T,k_pad,c]``   — per-(token, slot) contribution rows;
  - ``dswa_contrib[B,T,n_win,c]`` — window contribution rows.

The contributions are then reduced to ``dK_comp`` / ``dK_swa`` by an XLA
scatter-add over the gather indices. This is the deterministic-reduction
shape the paper prescribes for sparse-attention backward ("separate
accumulation buffers + a global deterministic summation"): every
contribution has exactly one producer, and XLA's scatter-add applies
updates in a fixed order — bit-reproducible across runs, no atomics.

HBM accounting of the bwd: re-gather reads (k+n_win)·c·2 B/token, plus a
write+read of the same volume for the contribution buffers. A
sort-by-destination two-pass scheme could remove the contribution
round-trip; it is the documented next step, not done here.

dsink is closed-form in XLA (as in kernel_v1):
``dsink_h = −Σ_{b,t} exp(sink_h − lse) · D``.
"""

from __future__ import annotations

from functools import lru_cache, partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np

from .config import ServingTiles


def _segment_reduce_sorted(dest, contrib, n_dest):
    """Deterministic dK reduction by sort-by-destination + segment-sum.

    ``dest`` [B, N] int32 (-1 = invalid), ``contrib`` [B, N, c] f32 →
    ``[B, n_dest, c]`` f32, where output row r is the sum of every
    contribution whose destination is r. Sorting by destination makes the
    same-row contributions contiguous, so the reduce is a single sorted
    ``segment_sum`` (a scan) rather than a collision-serialized
    scatter-add — fixed order either way (bit-reproducible), but the TPU
    write-conflict serialization on hot destinations is removed. Invalid
    (-1) entries route to a dump segment that is dropped.
    """
    B, N, c = contrib.shape
    valid = dest >= 0
    # Global segment id per (batch, entry); invalids -> the per-batch dump
    # row n_dest. Batches occupy disjoint [b*(n_dest+1), ...) ranges so one
    # flat segment_sum handles all of them.
    g = (jnp.where(valid, dest, n_dest)
         + jnp.arange(B, dtype=jnp.int32)[:, None] * (n_dest + 1)).reshape(-1)
    cf = contrib.reshape(B * N, c)
    order = jnp.argsort(g)                              # stable, deterministic
    seg = jax.ops.segment_sum(
        cf[order], g[order], num_segments=B * (n_dest + 1),
        indices_are_sorted=True)
    return seg.reshape(B, n_dest + 1, c)[:, :n_dest]


# ---------------------------------------------------------------------------
# Shared gather plumbing (single bf16 K array per cache)
# ---------------------------------------------------------------------------


def _row_copies(idx_smem, base, chunk, slot, hbm, buf, sem, b):
    return [
        pltpu.make_async_copy(
            hbm.at[b, jnp.maximum(idx_smem[0, 0, base + i], 0)],
            buf.at[slot, i], sem)
        for i in range(chunk)
    ]


def _swa_copy(hbm, buf, sem, b, start, n_win):
    return pltpu.make_async_copy(hbm.at[b, pl.ds(start, n_win)], buf, sem)


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------


def _fwd_kernel(
    idx_smem_ref,      # [1, 1, k_pad] int32 (SMEM)
    pos_smem_ref,      # [1, 1] int32 (SMEM)
    q_ref,             # [1, 1, n_h, c] bf16
    idx_vmem_ref,      # [1, 1, k_pad] int32
    sink_ref,          # [n_h, 1] f32
    kc_hbm,            # [B, S_c, c] bf16 (ANY)
    sw_hbm,            # [B, S_r, c] bf16 (ANY)
    o_ref,             # [1, 1, n_h, c] bf16
    lse_ref,           # [1, 1, n_h, 1] f32
    kbuf,              # [2, chunk, c]
    swbuf,             # [n_win, c]
    gsem, ssem,
    *,
    k_pad: int, chunk: int, n_win: int, s_raw: int, scale: float,
    phi_shift: float,
):
    b = pl.program_id(0)
    n_chunks = k_pad // chunk
    pos = pos_smem_ref[0, 0]
    start = jnp.clip(pos - n_win + 1, 0, s_raw - n_win)

    _swa_copy(sw_hbm, swbuf, ssem, b, start, n_win).start()
    for cp in _row_copies(idx_smem_ref, 0, chunk, 0, kc_hbm, kbuf, gsem, b):
        cp.start()

    q = q_ref[0, 0].astype(jnp.bfloat16)                   # [n_h, c]
    n_h = q.shape[0]
    l_i = jnp.zeros((n_h, 1), jnp.float32)
    acc = jnp.zeros(q.shape, jnp.float32)

    def update(l_i, acc, k_rows, valid):
        # Decoupled phi-softmax (constant shift C=phi_shift). lse emitted below
        # is exactly C + log(denom) = true sink-inclusive log-sum-exp, so the
        # backward (which recomputes p = exp(logit − lse)) is unchanged.
        logits = jax.lax.dot_general(
            q, k_rows, (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32) * jnp.float32(scale)
        phi = jnp.where(valid, jnp.exp(logits - jnp.float32(phi_shift)), 0.0)
        l_new = l_i + phi.sum(-1, keepdims=True)
        acc = acc + jax.lax.dot_general(
            phi.astype(jnp.bfloat16), k_rows, (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32)
        return l_new, acc

    for j in range(n_chunks):
        slot = j % 2
        if j + 1 < n_chunks:
            for cp in _row_copies(idx_smem_ref, (j + 1) * chunk, chunk,
                                  1 - slot, kc_hbm, kbuf, gsem, b):
                cp.start()
        for cp in _row_copies(idx_smem_ref, j * chunk, chunk, slot,
                              kc_hbm, kbuf, gsem, b):
            cp.wait()
        idx_vec = idx_vmem_ref[0, 0, j * chunk:(j + 1) * chunk]
        valid = (idx_vec >= 0).reshape(1, chunk)
        l_i, acc = update(l_i, acc, kbuf[slot].astype(jnp.bfloat16), valid)

    _swa_copy(sw_hbm, swbuf, ssem, b, start, n_win).wait()
    w_pos = start + jax.lax.broadcasted_iota(jnp.int32, (1, n_win), 1)
    valid = jnp.logical_and(w_pos <= pos, w_pos > pos - n_win)
    l_i, acc = update(l_i, acc, swbuf[...].astype(jnp.bfloat16), valid)

    sink = sink_ref[...]
    denom = l_i + jnp.exp(sink - jnp.float32(phi_shift))
    o_ref[0, 0] = (acc / denom).astype(o_ref.dtype)
    lse_ref[0, 0] = (jnp.float32(phi_shift) + jnp.log(denom)).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------


def _bwd_kernel(
    idx_smem_ref,      # [1, 1, k_pad] int32 (SMEM)
    pos_smem_ref,      # [1, 1] int32 (SMEM)
    q_ref,             # [1, 1, n_h, c] bf16
    idx_vmem_ref,      # [1, 1, k_pad] int32
    lse_ref,           # [1, 1, n_h, 1] f32
    d_ref,             # [1, 1, n_h, 1] f32     D = Σ_c dout·o
    dout_ref,          # [1, 1, n_h, c] bf16
    kc_hbm,            # [B, S_c, c] bf16 (ANY)
    sw_hbm,            # [B, S_r, c] bf16 (ANY)
    dq_ref,            # [1, 1, n_h, c] bf16
    dkc_ref,           # [1, 1, k_pad, c] bf16   per-token contributions
    dsw_ref,           # [1, 1, n_win, c] bf16
    kbuf, swbuf, gsem, ssem,
    *,
    k_pad: int, chunk: int, n_win: int, s_raw: int, scale: float,
):
    b = pl.program_id(0)
    n_chunks = k_pad // chunk
    pos = pos_smem_ref[0, 0]
    start = jnp.clip(pos - n_win + 1, 0, s_raw - n_win)

    _swa_copy(sw_hbm, swbuf, ssem, b, start, n_win).start()
    for cp in _row_copies(idx_smem_ref, 0, chunk, 0, kc_hbm, kbuf, gsem, b):
        cp.start()

    q = q_ref[0, 0].astype(jnp.bfloat16)                   # [n_h, c]
    dout = dout_ref[0, 0].astype(jnp.bfloat16)
    lse = lse_ref[0, 0]                                    # [n_h, 1]
    D = d_ref[0, 0]                                        # [n_h, 1]
    dq_acc = jnp.zeros(q.shape, jnp.float32)

    def grads(k_rows, valid):
        """(dq_contrib [n_h, c], dk_rows [rows, c]) for one key block."""
        logits = jax.lax.dot_general(
            q, k_rows, (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32) * jnp.float32(scale)
        p = jnp.where(valid, jnp.exp(logits - lse), 0.0)   # [n_h, rows]
        dp = jax.lax.dot_general(
            dout, k_rows, (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32)            # [n_h, rows]
        dlog = (p * (dp - D)).astype(jnp.bfloat16)         # [n_h, rows]
        dq_c = jax.lax.dot_general(
            dlog, k_rows, (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32) * jnp.float32(scale)
        # dk = scale·dlogᵀ·q + pᵀ·dout, contracting over heads.
        dk = jax.lax.dot_general(
            dlog, q, (((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32) * jnp.float32(scale)
        dk += jax.lax.dot_general(
            p.astype(jnp.bfloat16), dout, (((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32)            # [rows, c]
        return dq_c, dk

    for j in range(n_chunks):
        slot = j % 2
        if j + 1 < n_chunks:
            for cp in _row_copies(idx_smem_ref, (j + 1) * chunk, chunk,
                                  1 - slot, kc_hbm, kbuf, gsem, b):
                cp.start()
        for cp in _row_copies(idx_smem_ref, j * chunk, chunk, slot,
                              kc_hbm, kbuf, gsem, b):
            cp.wait()
        idx_vec = idx_vmem_ref[0, 0, j * chunk:(j + 1) * chunk]
        valid = (idx_vec >= 0).reshape(1, chunk)
        dq_c, dk = grads(kbuf[slot].astype(jnp.bfloat16), valid)
        dq_acc += dq_c
        dkc_ref[0, 0, j * chunk:(j + 1) * chunk, :] = dk.astype(dkc_ref.dtype)

    _swa_copy(sw_hbm, swbuf, ssem, b, start, n_win).wait()
    w_pos = start + jax.lax.broadcasted_iota(jnp.int32, (1, n_win), 1)
    valid = jnp.logical_and(w_pos <= pos, w_pos > pos - n_win)
    dq_c, dk = grads(swbuf[...].astype(jnp.bfloat16), valid)
    dq_acc += dq_c
    dsw_ref[0, 0] = dk.astype(dsw_ref.dtype)

    dq_ref[0, 0] = dq_acc.astype(dq_ref.dtype)


# ---------------------------------------------------------------------------
# pallas_call wrappers
# ---------------------------------------------------------------------------


def _common_in_specs(k_pad, n_h, c):
    return [
        pl.BlockSpec((1, 1, k_pad), lambda b, t: (b, t, 0),
                     memory_space=pltpu.SMEM),
        pl.BlockSpec((1, 1), lambda b, t: (b, t), memory_space=pltpu.SMEM),
        pl.BlockSpec((1, 1, n_h, c), lambda b, t: (b, t, 0, 0)),
        pl.BlockSpec((1, 1, k_pad), lambda b, t: (b, t, 0)),
    ]


def _fwd_call(q, kc, idx, swa, pos, sink2d, *, n_win, chunk, interpret):
    B, T, n_h, c = q.shape
    k_pad = idx.shape[-1]
    s_raw = swa.shape[1]
    scale = float(c) ** -0.5
    return pl.pallas_call(
        partial(_fwd_kernel, k_pad=k_pad, chunk=chunk, n_win=n_win,
                s_raw=s_raw, scale=scale, phi_shift=float(c) ** 0.5),
        grid=(B, T),
        in_specs=_common_in_specs(k_pad, n_h, c) + [
            pl.BlockSpec((n_h, 1), lambda b, t: (0, 0)),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[
            pl.BlockSpec((1, 1, n_h, c), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, 1, n_h, 1), lambda b, t: (b, t, 0, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((B, T, n_h, c), jnp.bfloat16),
            jax.ShapeDtypeStruct((B, T, n_h, 1), jnp.float32),
        ],
        scratch_shapes=[
            pltpu.VMEM((2, chunk, c), kc.dtype),
            pltpu.VMEM((n_win, c), swa.dtype),
            pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(
            # Tokens are independent: disjoint per-(b, t) o/lse outputs, caches
            # read-only — "parallel" T lets megacore split prefill.
            dimension_semantics=("parallel", "parallel")),
        interpret=interpret,
    )(idx, pos, q, idx, sink2d, kc, swa)


def _bwd_call(q, kc, idx, swa, pos, lse, D, dout, *, n_win, chunk, interpret):
    B, T, n_h, c = q.shape
    k_pad = idx.shape[-1]
    s_raw = swa.shape[1]
    scale = float(c) ** -0.5
    return pl.pallas_call(
        partial(_bwd_kernel, k_pad=k_pad, chunk=chunk, n_win=n_win,
                s_raw=s_raw, scale=scale),
        grid=(B, T),
        in_specs=_common_in_specs(k_pad, n_h, c) + [
            pl.BlockSpec((1, 1, n_h, 1), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, 1, n_h, 1), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, 1, n_h, c), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=[
            pl.BlockSpec((1, 1, n_h, c), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, 1, k_pad, c), lambda b, t: (b, t, 0, 0)),
            pl.BlockSpec((1, 1, n_win, c), lambda b, t: (b, t, 0, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((B, T, n_h, c), jnp.bfloat16),
            jax.ShapeDtypeStruct((B, T, k_pad, c), jnp.bfloat16),
            jax.ShapeDtypeStruct((B, T, n_win, c), jnp.bfloat16),
        ],
        scratch_shapes=[
            pltpu.VMEM((2, chunk, c), kc.dtype),
            pltpu.VMEM((n_win, c), swa.dtype),
            pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA,
        ],
        compiler_params=pltpu.CompilerParams(
            # Each (b, t) program writes its OWN per-token contribution slabs
            # (dq/dkc/dsw at out index (b, t, ...)); the cross-token dK
            # reduction happens later in XLA (_segment_reduce_sorted), so there
            # is no in-kernel accumulation race — T is "parallel".
            dimension_semantics=("parallel", "parallel")),
        interpret=interpret,
    )(idx, pos, q, idx, lse, D, dout, kc, swa)


# ---------------------------------------------------------------------------
# custom_vjp surface
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _make_diffable(n_win: int, chunk: int, interpret: bool):
    def _fwd_compute(q, kc, swa, sink, idx, pos):
        sink2d = sink.astype(jnp.float32).reshape(-1, 1)
        return _fwd_call(q.astype(jnp.bfloat16), kc, idx, swa, pos, sink2d,
                         n_win=n_win, chunk=chunk, interpret=interpret)

    @jax.custom_vjp
    def attn(q, kc, swa, sink, idx, pos):
        return _fwd_compute(q, kc, swa, sink, idx, pos)[0]

    def attn_fwd(q, kc, swa, sink, idx, pos):
        o, lse = _fwd_compute(q, kc, swa, sink, idx, pos)
        return o, (q, kc, swa, sink, idx, pos, o, lse)

    def attn_bwd(res, dout):
        q, kc, swa, sink, idx, pos, o, lse = res
        B, T, n_h, c = q.shape
        s_c, s_r = kc.shape[1], swa.shape[1]

        D = (dout.astype(jnp.float32) * o.astype(jnp.float32)).sum(
            -1, keepdims=True)                              # [B, T, n_h, 1]

        dq, dkc_contrib, dsw_contrib = _bwd_call(
            q.astype(jnp.bfloat16), kc, idx, swa, pos, lse, D,
            dout.astype(jnp.bfloat16),
            n_win=n_win, chunk=chunk, interpret=interpret)

        # Deterministic reductions, sort-by-destination (paper §3.3 fixed-
        # order accumulation). Sorting the per-(token,slot) contributions by
        # their destination K-row turns the reduction from a duplicate-index
        # scatter-add (write-conflict-serialized on TPU) into one sorted
        # segment-sum (a sequential scan) — same deterministic guarantee,
        # the destination-collision serialization gone. The contribution
        # buffer itself is still materialized; killing it outright needs a
        # destination-keyed recompute kernel that re-reads each source
        # token's q/dout ([n_h·c]) per pair instead of the reduced [c]
        # contribution — n_h× more source traffic for the memory saving,
        # not pursued (HARDWARE_NOTES §13.12).
        valid_k = idx >= 0                                  # [B, T, k_pad]
        contrib = dkc_contrib.astype(jnp.float32) * valid_k[..., None]
        dkc = _segment_reduce_sorted(
            jnp.where(valid_k, idx, -1).reshape(B, -1),
            contrib.reshape(B, -1, c), s_c)

        start = jnp.clip(pos - n_win + 1, 0, s_r - n_win)   # [B, T]
        w_pos = start[..., None] + jnp.arange(n_win, dtype=jnp.int32)
        valid_w = (w_pos <= pos[..., None]) & (w_pos > pos[..., None] - n_win)
        contrib_w = dsw_contrib.astype(jnp.float32) * valid_w[..., None]
        dsw = _segment_reduce_sorted(
            jnp.where(valid_w, w_pos, -1).reshape(B, -1),
            contrib_w.reshape(B, -1, c), s_r)

        # dsink (closed form; sink contributes only through lse).
        sink_b = sink.astype(jnp.float32)[None, None, :, None]
        dsink = (-jnp.exp(sink_b - lse) * D).sum(axis=(0, 1, 3))

        dgi = np.zeros(idx.shape, jax.dtypes.float0)
        dgp = np.zeros(pos.shape, jax.dtypes.float0)
        return (dq.astype(q.dtype), dkc.astype(kc.dtype),
                dsw.astype(swa.dtype), dsink.astype(sink.dtype), dgi, dgp)

    attn.defvjp(attn_fwd, attn_bwd)
    return attn


def sparse_mqa_train(
    q: jax.Array,            # [B, T, n_h, c]
    k_comp: jax.Array,       # [B, S_c, c] bf16 (roped + normed)
    topk_idxs: jax.Array,    # [B, T, k] int32, -1 invalid
    k_swa: jax.Array,        # [B, S_r, c] bf16
    q_pos: jax.Array,        # [B, T] int32
    attn_sink: jax.Array,    # [n_h]
    *,
    n_win: int,
    tiles: ServingTiles | None = None,
) -> jax.Array:
    """Differentiable gather attention (training path, bf16 KV).

    Gradients: dq, dK_comp, dK_swa (deterministic scatter-add over the
    per-token contributions), dsink. ``topk_idxs`` / ``q_pos`` get float0
    cotangents. K_full is never materialized in either direction.
    """
    if tiles is None:
        from .config import tiles_for
        tiles = tiles_for()
    chunk = tiles.attn_chunk
    k = topk_idxs.shape[-1]
    k_pad = ((k + chunk - 1) // chunk) * chunk
    if k_pad != k:
        topk_idxs = jnp.pad(topk_idxs, ((0, 0), (0, 0), (0, k_pad - k)),
                            constant_values=-1)
    fn = _make_diffable(n_win, chunk, tiles.interpret)
    return fn(q, k_comp.astype(jnp.bfloat16), k_swa.astype(jnp.bfloat16),
              attn_sink, topk_idxs.astype(jnp.int32), q_pos.astype(jnp.int32))


# ---------------------------------------------------------------------------
# Dense fp32 oracle (same semantics; jax.grad of this grades the bwd)
# ---------------------------------------------------------------------------


def sparse_mqa_train_ref(q, k_comp, topk_idxs, k_swa, q_pos, attn_sink, *,
                         n_win: int):
    B, T, n_h, c = q.shape
    scale = float(c) ** -0.5
    kcf = k_comp.astype(jnp.float32)
    krf = k_swa.astype(jnp.float32)
    qf = q.astype(jnp.float32)

    safe = jnp.maximum(topk_idxs, 0)
    K_sel = jax.vmap(lambda kb, ib: kb[ib])(kcf, safe)      # [B,T,k,c]
    sel_valid = topk_idxs >= 0

    start = jnp.clip(q_pos - n_win + 1, 0, krf.shape[1] - n_win)
    w_pos = start[..., None] + jnp.arange(n_win)
    K_win = jax.vmap(lambda kb, ib: kb[ib])(krf, w_pos)
    win_valid = (w_pos <= q_pos[..., None]) & (w_pos > q_pos[..., None] - n_win)

    K_all = jnp.concatenate([K_sel, K_win], axis=2)
    valid = jnp.concatenate([sel_valid, win_valid], axis=2)

    logits = jnp.einsum("bthc,btsc->bths", qf, K_all) * scale
    logits = jnp.where(valid[:, :, None, :], logits, -jnp.inf)
    m = jnp.maximum(jnp.max(logits, -1, keepdims=True),
                    attn_sink.astype(jnp.float32)[None, None, :, None])
    p = jnp.where(valid[:, :, None, :], jnp.exp(logits - m), 0.0)
    denom = p.sum(-1, keepdims=True) + jnp.exp(
        attn_sink.astype(jnp.float32)[None, None, :, None] - m)
    return jnp.einsum("bths,btsc->bthc", p / denom, K_all)
