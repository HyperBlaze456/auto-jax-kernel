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

What v1 does NOT do (TODO for v2+)
----------------------------------

  - Scalar-prefetched gather inside the kernel (kernel_refs §D4). Today the
    gather of ``K_comp[topk_idxs]`` materializes ``K_full[B, n, S, c]``
    via ``jnp.take_along_axis``, which is identical to the reference's
    bandwidth profile. v2 should pass ``topk_idxs`` through
    ``pltpu.PrefetchScalarGridSpec`` and gather inside.
  - Hand-written backward with saved (m, l) residuals (kernel_refs §D3).
    v1 uses ``jax.vjp`` through the reference forward.
  - Megacore-aware pipelining for the inner K-loop. We mark the outer
    (B, q-tile) axes as ``"parallel"`` — that already lets the compiler
    split across cores on v5p/v6p. The k-step axis accumulates and stays
    sequential. v2 should explore ``pltpu.emit_pipeline`` for inner
    multi-buffering.
  - Autotuned block-size search. ``config_for`` picks the largest tiles
    that fit the budget; v2 can layer a timing sweep on top, caching to
    disk keyed by ``(tpu, n_h, c, S)``.

To enable in dsv4/kernel.py: replace the ``ref.sparse_attn_with_sink`` call
in ``_sparse_attn_fwd`` with ``sparse_attn_kernel_v1`` from this module.
"""

from __future__ import annotations

from functools import lru_cache, partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from . import reference as ref
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
) -> jax.Array:
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

    return pl.pallas_call(
        partial(_flash_sink_kernel, nsteps=nsteps, scale=scale),
        grid=grid,
        in_specs=[q_spec, k_spec, mask_spec, sink_spec],
        out_specs=o_spec,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
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
# Surface entrypoint: gather + Pallas attention + sink
# ---------------------------------------------------------------------------

def _gather_concat_mask(q, K_comp, topk_idxs, K_swa):
    """The gather-and-concat preamble (same shape as ``ref.sparse_attn_with_sink``).

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

    scale = float(c) ** -0.5
    out = _flash_attn_with_sink_pallas(
        q, K_full, mask, attn_sink, scale=scale, config=config
    )
    if pad_n:
        out = out[:, :n]
    return out


@lru_cache(maxsize=None)
def _make_v1_kernel(config: KernelConfig):
    """Build a ``custom_vjp``-wrapped kernel specialized for ``config``.

    Cached on the (frozen, hashable) config so repeat calls with the same
    config reuse one closure — that's what keeps the JIT trace cache hot
    across calls. Each distinct config gets its own specialized fn.
    """

    @jax.custom_vjp
    def fn(q, K_comp, topk_idxs, K_swa, attn_sink):
        return _v1_forward(q, K_comp, topk_idxs, K_swa, attn_sink, config=config)

    def _fwd(q, K_comp, topk_idxs, K_swa, attn_sink):
        out = _v1_forward(q, K_comp, topk_idxs, K_swa, attn_sink, config=config)
        return out, (q, K_comp, topk_idxs, K_swa, attn_sink)

    def _bwd(res, dout):
        # v2: replace with a hand-written Pallas backward (kernel_refs.md §D3),
        # accumulating KV grads into per-program deterministic scratch
        # (paper §3.3).
        q, K_comp, topk_idxs, K_swa, attn_sink = res
        _, vjp_fn = jax.vjp(
            ref.sparse_attn_with_sink, q, K_comp, topk_idxs, K_swa, attn_sink
        )
        return vjp_fn(dout)

    fn.defvjp(_fwd, _bwd)
    return fn


def sparse_attn_kernel_v1(
    q: jax.Array,
    K_comp: jax.Array,
    topk_idxs: jax.Array,
    K_swa: jax.Array,
    attn_sink: jax.Array,
    *,
    config: KernelConfig | None = None,
) -> jax.Array:
    """V1 Pallas forward for the CSA sparse-MQA core.

    Surface matches ``dsv4.kernel.sparse_attn_kernel`` exactly so call sites
    don't change. Backward defers to autodiff through the reference impl;
    that's wrong for performance but correct for gradients, and gives v2
    something concrete to beat.

    ``config`` controls block sizes & lane width. When ``None`` (the
    default), the local TPU generation is auto-detected via
    ``dsv4.kernel_config.default_config``. Pass an explicit
    ``KernelConfig`` to pin a tuning or sweep one externally.
    """
    if config is None:
        _, _, n_h, c = q.shape
        config = default_config(n_h=n_h, c=c)
    return _make_v1_kernel(config)(q, K_comp, topk_idxs, K_swa, attn_sink)
