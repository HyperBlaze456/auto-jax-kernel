"""Pallas-TPU kernel surface for DeepSeek V4 hybrid attention.

THIS IS THE PUBLIC SURFACE THE BENCH + AGENT LOOP CONSUMES.

Surface contract (must not change):
  - csa_forward_kernel(H, params, cfg)         -> [B, n, n_h, c]
  - hca_forward_kernel(H, params, cfg)         -> [B, n, n_h, c]
  - mhc_sinkhorn_kernel(mixes, scale, base, hc, iters, eps) -> (pre, post, comb)
  - sparse_attn_kernel(q, K_comp, topk_idxs, K_swa, sink)  -> [B, n, n_h, c]

Routing
-------

The current default dispatches every entrypoint to ``dsv4.kernel_v2``,
which provides:

  - sparse_attn_kernel_v2: V1's FlashAttention-with-sink Pallas kernel
    (generation-agnostic via dsv4.kernel_config).
  - csa_forward_kernel_v2: full CSA forward (JAX preamble + Pallas core).
  - hca_forward_kernel_v2: full HCA forward (same Pallas core via
    select-all + block-causal-mask, mirroring reference.hca_forward).
  - mhc_sinkhorn_kernel_v2: per-token Pallas sinkhorn iteration loop.

Why no custom_vjp at this layer
-------------------------------

V2's inner ``_make_*_kernel`` factories already wire ``jax.custom_vjp``
around the diff-able array inputs, with non-diff Python primitives
(``hc``, ``sinkhorn_iters``, ``eps``, configs) captured via closure
instead of leaking through the custom_vjp boundary. Wrapping a SECOND
custom_vjp at this layer would coerce those Python primitives into
tracers (``DynamicJaxprTracer``) and then fail at any concrete Python
use downstream (slicing, dataclass field access, ``lru_cache`` key
lookup). The reference path's natural autodiff works the same way.

Backward defers to autodiff through the eager reference for all four
ops in V2. Hand-written Pallas backwards (kernel_refs.md §D3) are V3+
work.

Set ``DSV4_KERNEL=ref`` in the env to bypass V2 and route every
entrypoint to the eager reference (useful for sanity-checking the
backward path or for hosts without Pallas-TPU support).
"""

from __future__ import annotations

import os

import jax

from . import kernel_v2, reference as ref


# ``DSV4_KERNEL=ref`` forces the reference path (eager JAX, no Pallas).
# Anything else routes through V2.
_USE_REF = os.environ.get("DSV4_KERNEL", "v2").lower() == "ref"


# ---------------------------------------------------------------------------
# Sparse attention (the core CSA MQA stage)
# ---------------------------------------------------------------------------

def sparse_attn_kernel(
    q: jax.Array,
    K_comp: jax.Array,
    topk_idxs: jax.Array,
    K_swa: jax.Array,
    attn_sink: jax.Array,
) -> jax.Array:
    if _USE_REF:
        return ref.sparse_attn_with_sink(q, K_comp, topk_idxs, K_swa, attn_sink)
    return kernel_v2.sparse_attn_kernel_v2(q, K_comp, topk_idxs, K_swa, attn_sink)


# ---------------------------------------------------------------------------
# CSA / HCA full forward (compressor + indexer + MQA composed)
# ---------------------------------------------------------------------------

def csa_forward_kernel(H: jax.Array, params: ref.CSAParams, cfg: ref.CSAConfig) -> jax.Array:
    """End-to-end CSA forward.

    JAX-handled preamble (compressor, indexer, top-k, RoPE, RMSNorm) +
    Pallas FlashAttention core. The compressor and indexer stay in JAX
    because XLA's GEMM lowering is already good for those shapes; the
    win is replacing the eager [B, n, n_h, S] attention softmax with
    FlashAttention's online (m, l) state.
    """
    if _USE_REF:
        with jax.named_scope("csa_forward_ref"):
            return ref.csa_forward(H, params, cfg)
    with jax.named_scope("csa_forward"):
        return kernel_v2.csa_forward_kernel_v2(H, params, cfg)


def hca_forward_kernel(H: jax.Array, params: ref.HCAParams, cfg: ref.HCAConfig) -> jax.Array:
    """End-to-end HCA forward.

    Reuses the same Pallas sparse-MQA kernel as CSA, with a select-all
    block-causal mask in place of top-k indices. Errors at n_blk > 4096
    (same as the reference) until a streaming Pallas variant lands.
    """
    if _USE_REF:
        with jax.named_scope("hca_forward_ref"):
            return ref.hca_forward(H, params, cfg)
    with jax.named_scope("hca_forward"):
        return kernel_v2.hca_forward_kernel_v2(H, params, cfg)


# ---------------------------------------------------------------------------
# mHC Sinkhorn
# ---------------------------------------------------------------------------

def mhc_sinkhorn_kernel(
    mixes: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    hc: int,
    sinkhorn_iters: int,
    eps: float,
):
    """mHC Sinkhorn projection. ``hc``, ``sinkhorn_iters``, ``eps`` are
    Python primitives — they bind into the kernel closure rather than
    being treated as JAX arrays (see module docstring on custom_vjp).
    """
    if _USE_REF:
        return ref.mhc_sinkhorn(mixes, hc_scale, hc_base, hc, sinkhorn_iters, eps)
    return kernel_v2.mhc_sinkhorn_kernel_v2(
        mixes, hc_scale, hc_base, hc, sinkhorn_iters, eps,
    )
