"""Pallas-TPU kernel surface for DeepSeek V4 hybrid attention.

THIS IS THE FILE THE AGENT EDITS.

Surface contract (must not change):
  - csa_forward_kernel(H, params, cfg)         -> [B, n, n_h, c]
  - hca_forward_kernel(H, params, cfg)         -> [B, n, n_h, c]
  - mhc_sinkhorn_kernel(mixes, scale, base, hc, iters, eps) -> (pre, post, comb)
  - sparse_attn_kernel(q, K_comp, topk_idxs, K_swa, sink)  -> [B, n, n_h, c]

Each entrypoint is wired through jax.custom_vjp so the agent can hand-write
forward + backward Pallas kernels without changing call sites.

V1 baseline: every entrypoint dispatches to the eager reference (correct but
slow). The agent's job is to progressively replace each fwd/bwd with a
@pl.pallas_call kernel, gated on correctness vs reference (bench.py).

Tips for kernel work on Pallas-TPU:
  - jax.experimental.pallas as pl, jax.experimental.pallas.tpu as pltpu
  - Use BlockSpec to tile (B, n, n_h) over VMEM; sequence/topk dim becomes the
    K-loop you accumulate over with running (max, sum, output) à la FlashAttention.
  - For sparse_attn_kernel, the dominant op is gather-then-MQA: prefer to
    block over (n, k+n_win) and pull K_comp into VMEM via the topk_idxs scatter
    indirection (mirroring the reference TileLang sparse_attn_kernel block=64).
  - Attention sink + RoPE-on-output cleanup happen in a small post-pass; cheap.
  - For Sinkhorn, parallelize over (B, n) and keep the hc×hc matrix in registers.

Profiling:
  - Wrap the bench step in jax.named_scope("csa_fwd") etc. so XProf traces are
    legible. Use TPU_PROFILE_TRACE_DIR + jax.profiler.start_trace.
  - HLO dump: XLA_FLAGS="--xla_dump_to=/tmp/dsv4_hlo --xla_dump_hlo_as_text".
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import reference as ref


# ---------------------------------------------------------------------------
# Sparse attention (the core CSA MQA stage; matches reference sparse_attn_kernel)
# ---------------------------------------------------------------------------

@jax.custom_vjp
def sparse_attn_kernel(
    q: jax.Array,
    K_comp: jax.Array,
    topk_idxs: jax.Array,
    K_swa: jax.Array,
    attn_sink: jax.Array,
) -> jax.Array:
    """Forward of CSA's sparse MQA core. Replace body with Pallas kernel."""
    return ref.sparse_attn_with_sink(q, K_comp, topk_idxs, K_swa, attn_sink)


def _sparse_attn_fwd(q, K_comp, topk_idxs, K_swa, attn_sink):
    out = ref.sparse_attn_with_sink(q, K_comp, topk_idxs, K_swa, attn_sink)
    return out, (q, K_comp, topk_idxs, K_swa, attn_sink)


def _sparse_attn_bwd(res, dout):
    """Default backward via JAX autodiff through the reference impl.

    Agent: replace this with a hand-written Pallas backward (the conventional
    FlashAttention-2 backward, with KV-grad accumulation following the paper's
    deterministic per-SM accumulation buffer trick to avoid atomicAdd).
    """
    q, K_comp, topk_idxs, K_swa, attn_sink = res
    _, vjp_fn = jax.vjp(
        ref.sparse_attn_with_sink, q, K_comp, topk_idxs, K_swa, attn_sink
    )
    return vjp_fn(dout)


sparse_attn_kernel.defvjp(_sparse_attn_fwd, _sparse_attn_bwd)


# ---------------------------------------------------------------------------
# CSA / HCA full forward (compressor + indexer + MQA composed)
# ---------------------------------------------------------------------------

def csa_forward_kernel(H: jax.Array, params: ref.CSAParams, cfg: ref.CSAConfig) -> jax.Array:
    """End-to-end CSA. v1 is the reference; agent fuses parts into Pallas calls.

    Suggested fusion order (in priority for win-rate):
      1. sparse_attn_kernel (the inner MQA loop is the bandwidth bottleneck)
      2. csa_compress       (per-block softmax-mix; small reduction kernel)
      3. lightning indexer + top-k (reduce + selection in one pass)
    """
    with jax.named_scope("csa_forward"):
        return ref.csa_forward(H, params, cfg)


def hca_forward_kernel(H: jax.Array, params: ref.HCAParams, cfg: ref.HCAConfig) -> jax.Array:
    with jax.named_scope("hca_forward"):
        return ref.hca_forward(H, params, cfg)


# ---------------------------------------------------------------------------
# mHC Sinkhorn (matches reference hc_split_sinkhorn_kernel)
# ---------------------------------------------------------------------------

@jax.custom_vjp
def mhc_sinkhorn_kernel(
    mixes: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    hc: int,
    sinkhorn_iters: int,
    eps: float,
):
    return ref.mhc_sinkhorn(mixes, hc_scale, hc_base, hc, sinkhorn_iters, eps)


def _sink_fwd(mixes, hc_scale, hc_base, hc, sinkhorn_iters, eps):
    out = ref.mhc_sinkhorn(mixes, hc_scale, hc_base, hc, sinkhorn_iters, eps)
    return out, (mixes, hc_scale, hc_base, hc, sinkhorn_iters, eps)


def _sink_bwd(res, dout):
    mixes, hc_scale, hc_base, hc, sinkhorn_iters, eps = res
    _, vjp_fn = jax.vjp(
        lambda m, s, b: ref.mhc_sinkhorn(m, s, b, hc, sinkhorn_iters, eps),
        mixes, hc_scale, hc_base,
    )
    dm, ds, db = vjp_fn(dout)
    # Returns must align with primal arity, with None for non-diff (int / float).
    return dm, ds, db, None, None, None


mhc_sinkhorn_kernel.defvjp(_sink_fwd, _sink_bwd)
