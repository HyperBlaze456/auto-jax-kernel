"""V1 Pallas-TPU kernel for ``sparse_attn_kernel``.

This is the first hand-written Pallas pass over the CSA MQA core. It's a
faithful, conservative baseline meant to:

  - prove out the surface contract (correctness vs. the reference);
  - establish the FlashAttention-with-sink pattern we'll iterate on;
  - leave clear room for the v2/v3 optimizations called out below;
  - **serve every TPU generation from one source** by lifting block sizes
    and lane width into ``KernelConfig`` (see ``dsv4/kernel_config.py``).

Design choices for v1
---------------------

The reference ``sparse_attn_with_sink`` does three things: (1) gather the
top-k compressed K entries per query, (2) concatenate with the SWA window,
(3) attend with an online softmax + per-head sink. v1 keeps step (1)+(2) in
plain JAX (so we get a single dense ``K_full[B, n, S, c]`` tensor) and pulls
just step (3) into a Pallas kernel. That's enough to demonstrate the
FlashAttention pattern without also having to wire up scalar-prefetch
gather indices (deferred to v2 — kernel_refs.md §D4).

We use the V4 convention V == K (paper §2.3: "each compressed KV entry
serves as both attention key and value"), so only K_full is passed in.

Block shape rationale (kernel_refs.md §C / §J)
----------------------------------------------

Block sizes are now resolved per (TPU generation, problem shape) via
``dsv4.kernel_config.config_for``. The shapes below describe the *kind*
of tile; concrete numbers come from the resolved ``KernelConfig``.

  - q tile: ``(1, BQ, n_h, c)``. Last two dims ``(n_h, c)`` match the full
    array dims, which exempts them from the lane/sublane multiples rule.
    ``n_h`` is 64 (Flash) or 128 (Pro); ``c`` is 512. Both already aligned.
  - K tile: ``(1, BQ, BS, c)``. ``BS`` is a multiple of the VPU lane width
    (128 on every current generation; configurable via
    ``KernelConfig.lane_size`` for hypothetical future parts).
  - Scratch ``m, l, acc`` are stored at ``(1, BQ, n_h, c)`` fp32. We
    broadcast the per-(query, head) scalar across the c lane axis to
    sidestep the lane-multiple rule for tiny last dims (the natural shape
    for m/l is ``(BQ, n_h)`` which violates it). Memory cost is trivial
    relative to acc and keeps the layout uniform.

Numerics (kernel_refs.md §I)
----------------------------

  - bf16 inputs, fp32 accumulator. ``preferred_element_type=jnp.float32``
    on both QK^T and PV forces the bf16×bf16→f32 MXU path.
  - Online softmax keeps ``m, l`` in fp32.
  - Masked-position logit set to ``-1e30`` (large-negative-but-finite) so
    we never see ``-inf - -inf`` NaNs in ``alpha = exp(m_prev - m_new)`` on
    the first iteration when an entire chunk is masked. ``p`` is also
    explicitly masked to zero so the denominator stays exact.

Backward (kernel_refs §D3)
--------------------------

v1 ships a hand-written Pallas backward for the FlashAttention-with-sink
core. The forward kernel writes one extra HBM output, ``lse`` (the
combined log-sum-exp including the sink term), shape ``[B, n, n_h, 1]``
fp32. With that saved residual, the bwd kernel recomputes ``p`` from
``q · k^T`` and ``lse``, then accumulates:

  - ``dq`` across the k-step axis (per-program VMEM scratch, finalized at
    the last step) — same accumulation pattern as the forward acc.
  - ``dk`` per-tile (QK contribution + PV contribution; V == K under V4).
  - ``dsink`` is closed-form in JAX outside the kernel:
    ``dsink_h = -∑_(b,t) exp(sink_h - lse_{b,t,h}) · (dout · o)_{b,t,h}``.

The custom_vjp boundary is wrapped around just the Pallas call (not the
whole ``sparse_attn_kernel_v1`` surface). The gather/concat preamble
that builds ``K_full`` from ``K_comp`` and ``topk_idxs`` is plain JAX
and gets its bwd from natural autodiff — that's what produces dK_comp /
dK_swa from dK_full, including the scatter-add over duplicate top-k
selections.

What v1 still does NOT do (TODO for v2+)
----------------------------------------

  - Scalar-prefetched gather inside the kernel (kernel_refs §D4). Today the
    gather of ``K_comp[topk_idxs]`` materializes ``K_full[B, n, S, c]``
    via ``jnp.take_along_axis``, which is identical to the reference's
    bandwidth profile. v2+ should pass ``topk_idxs`` through
    ``pltpu.PrefetchScalarGridSpec`` and gather inside.
  - Per-program deterministic KV-grad scratch (paper §3.3). The current
    bwd writes ``dk`` per-tile, which is already deterministic on TPU
    because no two programs share the same (b, qi, si) slot. The §3.3
    pattern (per-SM accumulation buffer + global deterministic sum) only
    matters once we tile the same K entry across multiple programs —
    e.g., a streaming bwd over very long S.
  - Megacore-aware pipelining for the inner K-loop. We mark the outer
    (B, q-tile) axes as ``"parallel"`` — that already lets the compiler
    split across cores on v5p/v6p. The k-step axis accumulates and stays
    sequential. v2+ should explore ``pltpu.emit_pipeline`` for inner
    multi-buffering.
  - Autotuned block-size search. ``config_for`` picks the largest tiles
    that fit the budget; v2+ can layer a timing sweep on top, caching to
    disk keyed by ``(tpu, n_h, c, S)``.
"""

from __future__ import annotations

from functools import lru_cache, partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from .kernel_config import KernelConfig, default_config


# Large-negative-but-finite sentinel for masked logits. Avoids -inf-(-inf)
# NaNs in alpha when an entire K-block is fully masked at first touch.
_NEG_INF_F32 = -1.0e30


# ---------------------------------------------------------------------------
# Pallas kernel: FlashAttention forward with per-head attention sink
# ---------------------------------------------------------------------------

def _flash_sink_kernel(
    q_ref,           # [1, BQ, n_h, c]   bf16/f32  — queries
    k_ref,           # [1, BQ, BS, c]    bf16/f32  — keys (= values; V4)
    mask_ref,        # [1, BQ, BS]       bool      — validity
    sink_ref,        # [n_h]             f32       — per-head sink logits
    o_ref,           # [1, BQ, n_h, c]   bf16/f32  — output (written on last step)
    lse_ref,         # [1, BQ, n_h, 1]   f32       — combined log-sum-exp (written on last step)
    m_ref,           # [1, BQ, n_h, c]   f32 scratch — running max  (broadcast over c)
    l_ref,           # [1, BQ, n_h, c]   f32 scratch — running denom (broadcast over c)
    acc_ref,         # [1, BQ, n_h, c]   f32 scratch — running numerator
    *,
    nsteps: int,
    scale: float,
):
    s = pl.program_id(2)

    @pl.when(s == 0)
    def _init_scratch():
        m_ref[...] = jnp.full(m_ref.shape, _NEG_INF_F32, dtype=jnp.float32)
        l_ref[...] = jnp.zeros(l_ref.shape, dtype=jnp.float32)
        acc_ref[...] = jnp.zeros(acc_ref.shape, dtype=jnp.float32)

    q = q_ref[...].astype(jnp.float32)        # [1, BQ, n_h, c]
    k = k_ref[...].astype(jnp.float32)        # [1, BQ, BS, c]
    mask = mask_ref[...].astype(jnp.bool_)    # [1, BQ, BS]

    # Per-(b, t) MQA QK^T. Contract over c, batch over (b, t).
    # q: [1, BQ, n_h, c]  ·  k: [1, BQ, BS, c]  →  logits: [1, BQ, n_h, BS]
    logits = jax.lax.dot_general(
        q, k,
        dimension_numbers=(((3,), (3,)), ((0, 1), (0, 1))),
        preferred_element_type=jnp.float32,
    ) * jnp.float32(scale)

    mask_b = mask[:, :, None, :]                                     # [1, BQ, 1, BS]
    logits = jnp.where(mask_b, logits, jnp.float32(_NEG_INF_F32))

    # Read scratch; the per-(query, head) scalars live in lane 0 of the
    # broadcasted-over-c representation (see header on the layout choice).
    m_prev = m_ref[..., :1]                                          # [1, BQ, n_h, 1]
    l_prev = l_ref[..., :1]
    acc_prev = acc_ref[...]                                          # [1, BQ, n_h, c]

    m_curr = jnp.max(logits, axis=-1, keepdims=True)                 # [1, BQ, n_h, 1]
    m_new = jnp.maximum(m_prev, m_curr)

    p = jnp.exp(logits - m_new)                                      # [1, BQ, n_h, BS]
    p = jnp.where(mask_b, p, jnp.float32(0.0))                       # zero masked positions
    alpha = jnp.exp(m_prev - m_new)                                  # [1, BQ, n_h, 1]
    l_new = alpha * l_prev + jnp.sum(p, axis=-1, keepdims=True)

    # PV. p: [1, BQ, n_h, BS]  ·  k (=v): [1, BQ, BS, c]  →  [1, BQ, n_h, c].
    pv = jax.lax.dot_general(
        p, k,
        dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1))),
        preferred_element_type=jnp.float32,
    )
    acc_new = acc_prev * alpha + pv

    # Persist running state. m_new / l_new broadcast over the c lane axis.
    m_ref[...] = jnp.broadcast_to(m_new, m_ref.shape).astype(jnp.float32)
    l_ref[...] = jnp.broadcast_to(l_new, l_ref.shape).astype(jnp.float32)
    acc_ref[...] = acc_new

    @pl.when(s == nsteps - 1)
    def _finalize():
        m_final = m_ref[..., :1]                                     # [1, BQ, n_h, 1]
        l_final = l_ref[..., :1]
        acc_final = acc_ref[...]                                     # [1, BQ, n_h, c]
        sink = sink_ref[...].astype(jnp.float32)                     # [n_h]
        sink_b = sink[None, None, :, None]                           # [1, 1, n_h, 1]

        # Combine the running (m, l) with the sink: mathematically equivalent
        # to having a virtual logit equal to ``sink_h`` whose value vector is
        # zero (so it contributes only to the denominator).
        m_combined = jnp.maximum(m_final, sink_b)
        alpha_final = jnp.exp(m_final - m_combined)
        sink_term = jnp.exp(sink_b - m_combined)
        denom = l_final * alpha_final + sink_term
        out = (acc_final * alpha_final) / denom
        o_ref[...] = out.astype(o_ref.dtype)

        # lse = log-sum-exp over (valid logits ∪ sink), in the original logit
        # frame. Saved for the bwd kernel — recovers p_s = exp(s - lse) and
        # p_sink = exp(sink - lse) without re-running online softmax.
        lse = m_combined + jnp.log(denom)
        lse_ref[...] = lse.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Pallas-call wrapper
# ---------------------------------------------------------------------------

def _flash_attn_with_sink_pallas(
    q: jax.Array,         # [B, n, n_h, c]
    K_full: jax.Array,    # [B, n, S, c] (S already padded to a multiple of config.bs)
    mask: jax.Array,      # [B, n, S] bool
    attn_sink: jax.Array, # [n_h]
    scale: float,
    config: KernelConfig,
) -> tuple[jax.Array, jax.Array]:
    """Returns ``(out, lse)``. ``lse`` is the combined (sink-inclusive)
    log-sum-exp per (b, t, h); saved as a residual for the bwd kernel.
    """
    B, n, n_h, c = q.shape
    S = K_full.shape[2]
    bq, bs = config.bq, config.bs
    if S % bs != 0:
        raise ValueError(f"S={S} must be a multiple of BS={bs}; pad in the wrapper")
    if n % bq != 0:
        raise ValueError(f"n={n} must be a multiple of BQ={bq}; pad in the wrapper")
    if c % config.lane_size != 0:
        raise ValueError(
            f"head dim c={c} must be a multiple of lane_size={config.lane_size} "
            f"for VPU alignment"
        )

    nsteps = S // bs
    grid = (B, n // bq, nsteps)

    # The grid maps to (b, q_tile, k_step). Indices come back in that order
    # as the index_map's positional args.
    q_spec    = pl.BlockSpec((1, bq, n_h, c), lambda b, qi, si: (b, qi, 0, 0))
    k_spec    = pl.BlockSpec((1, bq, bs, c),  lambda b, qi, si: (b, qi, si, 0))
    mask_spec = pl.BlockSpec((1, bq, bs),     lambda b, qi, si: (b, qi, si))
    sink_spec = pl.BlockSpec((n_h,),          lambda b, qi, si: (0,))
    o_spec    = pl.BlockSpec((1, bq, n_h, c), lambda b, qi, si: (b, qi, 0, 0))
    # ``lse`` is per (b, t, h); the trailing-1 last dim is "natural" (matches
    # the original array's last dim), which exempts it from the lane-multiple
    # rule. The n_h dim likewise matches the original (sublane-rule exempt).
    lse_spec  = pl.BlockSpec((1, bq, n_h, 1), lambda b, qi, si: (b, qi, 0, 0))

    return pl.pallas_call(
        partial(_flash_sink_kernel, nsteps=nsteps, scale=scale),
        grid=grid,
        in_specs=[q_spec, k_spec, mask_spec, sink_spec],
        out_specs=[o_spec, lse_spec],
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct((B, n, n_h, 1), jnp.float32),
        ],
        scratch_shapes=[
            pltpu.VMEM((1, bq, n_h, c), jnp.float32),  # m
            pltpu.VMEM((1, bq, n_h, c), jnp.float32),  # l
            pltpu.VMEM((1, bq, n_h, c), jnp.float32),  # acc
        ],
        compiler_params=pltpu.CompilerParams(
            # (B, q-tile) are embarrassingly parallel — on megacore parts
            # (v5p/v6p) the compiler splits them across the two cores; on
            # single-core parts the annotation is a no-op. The k-step axis
            # accumulates and stays sequential.
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
        interpret=config.interpret,
    )(q, K_full, mask, attn_sink)


# ---------------------------------------------------------------------------
# Pallas kernel: FlashAttention backward via saved (lse) residual
# ---------------------------------------------------------------------------
#
# Given the forward's saved log-sum-exp ``lse`` (which already absorbs the
# sink term) and the precomputed scalar ``D = Σ_c dout · o``, we recover the
# attention weights inside the bwd as ``p_s = exp(s_s - lse) · mask`` and
# apply the standard softmax-backprop identity ``dlogits_s = p_s · (dp_s - D)``
# where ``dp_s = Σ_c dout · v_s`` (and v == k under V4). Because the sink has
# value vector 0, it only shifts ``lse`` — it never appears in ``D`` or in
# ``dp``, and the sink's own gradient is closed-form in JAX outside the kernel.
#
# Accumulation pattern, mirroring the forward:
#   - ``dq`` accumulates across the k-step axis (per-program VMEM scratch),
#     written to HBM at the last step.
#   - ``dk`` (= dv under V4) is per-(b, qi, si) tile: QK contribution
#     ``Σ_h scale · dlogits_s · q`` + PV contribution ``Σ_h p_s · dout``.
# No two grid programs write to the same HBM (b, qi, si) slot, so dk is
# deterministic without the per-SM-buffer pattern from paper §3.3.


def _flash_sink_kernel_bwd(
    q_ref,           # [1, BQ, n_h, c]   bf16/f32
    k_ref,           # [1, BQ, BS, c]    bf16/f32   (k == v in V4)
    mask_ref,        # [1, BQ, BS]       bool
    lse_ref,         # [1, BQ, n_h, 1]   f32
    D_ref,           # [1, BQ, n_h, 1]   f32        precomputed (Σ_c dout · o)
    dout_ref,        # [1, BQ, n_h, c]   bf16/f32
    dq_ref,          # [1, BQ, n_h, c]   bf16/f32   out (finalized at last step)
    dk_ref,          # [1, BQ, BS, c]    bf16/f32   out (per-tile)
    dq_scratch_ref,  # [1, BQ, n_h, c]   f32 scratch (running dq accumulator)
    *,
    nsteps: int,
    scale: float,
):
    s = pl.program_id(2)

    @pl.when(s == 0)
    def _init_dq():
        dq_scratch_ref[...] = jnp.zeros(dq_scratch_ref.shape, dtype=jnp.float32)

    q = q_ref[...].astype(jnp.float32)        # [1, BQ, n_h, c]
    k = k_ref[...].astype(jnp.float32)        # [1, BQ, BS, c]
    mask = mask_ref[...].astype(jnp.bool_)    # [1, BQ, BS]
    lse = lse_ref[...]                        # [1, BQ, n_h, 1]
    D = D_ref[...]                            # [1, BQ, n_h, 1]
    dout = dout_ref[...].astype(jnp.float32)  # [1, BQ, n_h, c]

    # Recompute attention weights p from the saved lse.
    logits = jax.lax.dot_general(
        q, k,
        dimension_numbers=(((3,), (3,)), ((0, 1), (0, 1))),
        preferred_element_type=jnp.float32,
    ) * jnp.float32(scale)                                    # [1, BQ, n_h, BS]

    mask_b = mask[:, :, None, :]                              # [1, BQ, 1, BS]
    p = jnp.exp(logits - lse)                                 # [1, BQ, n_h, BS]
    p = jnp.where(mask_b, p, jnp.float32(0.0))

    # dp[b,t,h,s] = Σ_c dout[b,t,h,c] · k[b,t,s,c]    (same contraction as fwd QK^T)
    dp = jax.lax.dot_general(
        dout, k,
        dimension_numbers=(((3,), (3,)), ((0, 1), (0, 1))),
        preferred_element_type=jnp.float32,
    )                                                          # [1, BQ, n_h, BS]

    # Standard softmax-bwd identity. Masked positions stay 0 because p is
    # zeroed for them, which propagates through the multiply.
    dlogits = p * (dp - D)                                     # [1, BQ, n_h, BS]

    # dq contribution from this s-tile.
    # dlogits: [..., n_h, BS] · k: [..., BS, c] → [..., n_h, c]
    dq_contrib = jax.lax.dot_general(
        dlogits, k,
        dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1))),
        preferred_element_type=jnp.float32,
    ) * jnp.float32(scale)                                     # [1, BQ, n_h, c]
    dq_scratch_ref[...] = dq_scratch_ref[...] + dq_contrib

    # dk per-tile, QK + PV. Both contract over n_h (dim 2 in both operands).
    dk_qk = jax.lax.dot_general(
        dlogits, q,
        dimension_numbers=(((2,), (2,)), ((0, 1), (0, 1))),
        preferred_element_type=jnp.float32,
    ) * jnp.float32(scale)                                     # [1, BQ, BS, c]
    dk_pv = jax.lax.dot_general(
        p, dout,
        dimension_numbers=(((2,), (2,)), ((0, 1), (0, 1))),
        preferred_element_type=jnp.float32,
    )                                                          # [1, BQ, BS, c]
    dk_ref[...] = (dk_qk + dk_pv).astype(dk_ref.dtype)

    @pl.when(s == nsteps - 1)
    def _finalize_dq():
        dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)


def _flash_attn_with_sink_pallas_bwd(
    q: jax.Array,         # [B, n, n_h, c]   (padded to BQ)
    K_full: jax.Array,    # [B, n, S, c]     (padded to BQ along n, BS along S)
    mask: jax.Array,      # [B, n, S] bool
    attn_sink: jax.Array, # [n_h]
    out: jax.Array,       # [B, n, n_h, c]   forward output (padded)
    lse: jax.Array,       # [B, n, n_h, 1] f32
    dout: jax.Array,      # [B, n, n_h, c]   upstream cotangent (padded)
    scale: float,
    config: KernelConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Bwd kernel call. Returns ``(dq, dk, dsink)``.

    ``dsink`` is computed entirely in JAX outside the Pallas grid because
    the sink term only contributes through ``lse`` — its gradient is
    ``dsink_h = -Σ_(b,t) exp(sink_h - lse_{b,t,h}) · D_{b,t,h}``.
    """
    B, n, n_h, c = q.shape
    S = K_full.shape[2]
    bq, bs = config.bq, config.bs
    nsteps = S // bs

    # Precompute D = Σ_c (dout · o). Shape [B, n, n_h, 1] f32, same layout
    # as lse so it tiles into the kernel under the same BlockSpec.
    D = (dout.astype(jnp.float32) * out.astype(jnp.float32)).sum(
        axis=-1, keepdims=True,
    )

    grid = (B, n // bq, nsteps)
    q_spec    = pl.BlockSpec((1, bq, n_h, c), lambda b, qi, si: (b, qi, 0, 0))
    k_spec    = pl.BlockSpec((1, bq, bs, c),  lambda b, qi, si: (b, qi, si, 0))
    mask_spec = pl.BlockSpec((1, bq, bs),     lambda b, qi, si: (b, qi, si))
    lse_spec  = pl.BlockSpec((1, bq, n_h, 1), lambda b, qi, si: (b, qi, 0, 0))
    d_spec    = pl.BlockSpec((1, bq, n_h, 1), lambda b, qi, si: (b, qi, 0, 0))
    dout_spec = pl.BlockSpec((1, bq, n_h, c), lambda b, qi, si: (b, qi, 0, 0))
    dq_spec   = pl.BlockSpec((1, bq, n_h, c), lambda b, qi, si: (b, qi, 0, 0))
    dk_spec   = pl.BlockSpec((1, bq, bs, c),  lambda b, qi, si: (b, qi, si, 0))

    dq, dk = pl.pallas_call(
        partial(_flash_sink_kernel_bwd, nsteps=nsteps, scale=scale),
        grid=grid,
        in_specs=[q_spec, k_spec, mask_spec, lse_spec, d_spec, dout_spec],
        out_specs=[dq_spec, dk_spec],
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(K_full.shape, K_full.dtype),
        ],
        scratch_shapes=[pltpu.VMEM((1, bq, n_h, c), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
        interpret=config.interpret,
    )(q, K_full, mask, lse, D, dout)

    # dsink, closed-form in JAX.
    sink_b = attn_sink[None, None, :, None].astype(jnp.float32)  # [1, 1, n_h, 1]
    p_sink = jnp.exp(sink_b - lse)                                # [B, n, n_h, 1]
    dsink = (-p_sink * D).sum(axis=(0, 1, 3)).astype(attn_sink.dtype)
    return dq, dk, dsink


# ---------------------------------------------------------------------------
# Surface entrypoint: gather + Pallas attention + sink
# ---------------------------------------------------------------------------

def _gather_concat_mask(q, K_comp, topk_idxs, K_swa):
    """The gather-and-concat preamble (same shape as ``eager.sparse_attn_with_sink``).

    Returns ``(K_full, mask)`` with K_full of shape ``[B, n, k+n_win, c]`` and a
    bool mask of the same leading shape. -1 padding entries in ``topk_idxs``
    are mapped to safe-index 0 and masked out.
    """
    B, n, _, c = q.shape
    n_win = K_swa.shape[2]

    safe_idx = jnp.where(topk_idxs < 0, 0, topk_idxs)                       # [B, n, k]
    K_sel = jnp.take_along_axis(
        K_comp[:, None, :, :].repeat(n, axis=1),                            # [B, n, n_blk, c]
        safe_idx[..., None].repeat(c, axis=-1),                             # [B, n, k, c]
        axis=2,
    )

    K_full = jnp.concatenate([K_sel, K_swa], axis=2)                        # [B, n, k+n_win, c]
    valid_topk = topk_idxs >= 0
    valid_swa = jnp.ones((B, n, n_win), dtype=jnp.bool_)
    mask = jnp.concatenate([valid_topk, valid_swa], axis=-1)                # [B, n, k+n_win]
    return K_full, mask


# ---------------------------------------------------------------------------
# Diffable Pallas boundary: forward + hand-written bwd, wrapped in custom_vjp
# ---------------------------------------------------------------------------
#
# This is the narrow boundary the custom_vjp lives around: the Pallas forward
# (which also emits ``lse``) and the Pallas backward (which consumes it).
# Everything outside — the gather/concat that builds K_full from K_comp and
# topk_idxs — stays in JAX, so natural autodiff handles it and produces
# dK_comp / dK_swa (including the scatter-add over duplicate top-k entries).
#
# Why the float-mask cast: ``mask`` is a bool tensor. ``jax.custom_vjp`` is
# strictest about cotangents matching primal dtypes, and bool primals do not
# carry a meaningful float cotangent. We pass the mask through the boundary
# as fp32 (cast once, ~negligible HBM cost), then re-binarize inside. The
# bwd returns ``jnp.zeros_like(mask_f)`` for that slot.


@lru_cache(maxsize=None)
def _make_flash_diffable(config: KernelConfig, c: int):
    """Cached ``custom_vjp`` closure for the Pallas flash core.

    Cached on ``(config, c)`` so the scale and block sizes are constants in
    the traced kernel (keeping the JIT cache hot across calls with the same
    shapes).
    """
    scale = float(c) ** -0.5

    @jax.custom_vjp
    def _flash(q, K_full, mask_f, attn_sink):
        mask_bool = mask_f > jnp.float32(0.5)
        out, _lse = _flash_attn_with_sink_pallas(
            q, K_full, mask_bool, attn_sink, scale=scale, config=config,
        )
        return out

    def _fwd(q, K_full, mask_f, attn_sink):
        mask_bool = mask_f > jnp.float32(0.5)
        out, lse = _flash_attn_with_sink_pallas(
            q, K_full, mask_bool, attn_sink, scale=scale, config=config,
        )
        return out, (q, K_full, mask_f, mask_bool, attn_sink, out, lse)

    def _bwd(res, dout):
        q, K_full, mask_f, mask_bool, attn_sink, out, lse = res
        dq, dk, dsink = _flash_attn_with_sink_pallas_bwd(
            q, K_full, mask_bool, attn_sink, out, lse, dout,
            scale=scale, config=config,
        )
        # Cotangents matching the four primal args: (q, K_full, mask_f, attn_sink).
        return dq, dk, jnp.zeros_like(mask_f), dsink

    _flash.defvjp(_fwd, _bwd)
    return _flash


def _v1_forward(q, K_comp, topk_idxs, K_swa, attn_sink, *, config: KernelConfig):
    B, n, n_h, c = q.shape
    K_full, mask = _gather_concat_mask(q, K_comp, topk_idxs, K_swa)

    # Pad ``n`` to a multiple of config.bq. Different TPU generations resolve
    # different BQ values, so we can't push this requirement onto the caller
    # without leaking generation-specific knowledge. Slice the padding off
    # the output before returning.
    pad_n = (-n) % config.bq
    if pad_n:
        q = jnp.pad(q, ((0, 0), (0, pad_n), (0, 0), (0, 0)))
        K_full = jnp.pad(K_full, ((0, 0), (0, pad_n), (0, 0), (0, 0)))
        mask = jnp.pad(mask, ((0, 0), (0, pad_n), (0, 0)), constant_values=False)

    # Pad the seq axis of K_full / mask to a multiple of config.bs so the
    # kernel's block grid aligns. mask=False keeps padded positions inert.
    S = K_full.shape[2]
    pad_s = (-S) % config.bs
    if pad_s:
        K_full = jnp.pad(K_full, ((0, 0), (0, 0), (0, pad_s), (0, 0)))
        mask = jnp.pad(mask, ((0, 0), (0, 0), (0, pad_s)), constant_values=False)

    # Cast bool→fp32 for the custom_vjp boundary, then call the diffable
    # closure. The bwd of the bool→fp32 cast is naturally zero, which is
    # correct (mask is a function of topk_idxs, which has no float grad).
    mask_f = mask.astype(jnp.float32)
    out = _make_flash_diffable(config, c)(q, K_full, mask_f, attn_sink)

    if pad_n:
        out = out[:, :n]
    return out


def sparse_attn_kernel_v1(
    q: jax.Array,
    K_comp: jax.Array,
    topk_idxs: jax.Array,
    K_swa: jax.Array,
    attn_sink: jax.Array,
    *,
    config: KernelConfig | None = None,
) -> jax.Array:
    """V1 Pallas forward+bwd for the CSA sparse-MQA core.

    Surface matches ``dsv4.kernel.sparse_attn_kernel`` exactly so call sites
    don't change. The Pallas forward emits ``lse`` as a saved residual; the
    custom_vjp boundary (around just the Pallas call, not the gather/concat)
    dispatches to the hand-written Pallas bwd that recomputes ``p`` from
    ``lse`` and accumulates ``dq`` / ``dk`` / ``dsink`` (see the bwd kernel
    docstring above).

    ``config`` controls block sizes & lane width. When ``None`` (the
    default), the local TPU generation is auto-detected via
    ``dsv4.kernel_config.default_config``. Pass an explicit
    ``KernelConfig`` to pin a tuning or sweep one externally.
    """
    if config is None:
        _, _, n_h, c = q.shape
        config = default_config(n_h=n_h, c=c)
    return _v1_forward(q, K_comp, topk_idxs, K_swa, attn_sink, config=config)
