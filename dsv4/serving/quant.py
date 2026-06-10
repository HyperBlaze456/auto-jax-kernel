"""FP8 block-scaled quantization primitives (DeepGEMM convention, TPU layout).

Conventions
-----------

All GEMM-facing activations are 2-D ``[M, K]`` (callers flatten leading
dims). Two quantization granularities, matching the DSv3/DSv4 FP8 framework:

  - **Activations**: per ``1 x 128`` group along K. Quantized payload
    ``q[M, K]`` (e4m3) + scales stored **transposed**: ``s_t[K/128, M]``
    fp32.
  - **Weights**: per ``128 x 128`` block. Payload ``q[K, N]`` (e4m3,
    k-major so the kernel's ``[tk, N]`` tile is a contiguous slab) +
    scales ``s[K/128, N/128]`` fp32. Grouped (MoE) weights add a leading
    expert dim: ``q[E, K, N]``, ``s[E, K/128, N/128]``.

Why transposed activation scales
--------------------------------

Inside a Pallas TPU kernel, a block's *last* dim must be a multiple of the
lane width (128) or cover the full array dim, and dynamic indexing is only
cheap on sublane (non-last) dims. With ``s_t[K/128, M]`` the kernel loads
the full-K scale strip for its m-tile as one rule-safe block
(``(K/128, tm)``, last dim 128-aligned) and indexes the k-block row
dynamically on the sublane axis. The same convention lets the fused
SwiGLU+quant epilogue *emit* scales already transposed for the next GEMM —
the scale tensor never gets relaid out in HBM between expert GEMM 1 and 2.

Numerics
--------

``scale = max(amax(group), eps) / 448``; payload is clamped to ±448 before
the cast so saturation is explicit (e4m3fn has no inf; XLA's overflow
behavior is NaN). Dequantization ``q * s`` in fp32. The quantize→dequantize
round-trip error is the *only* approximation in the FP8 path — every dot
downstream accumulates in fp32 and applies scales in fp32 (two-level
accumulation), so kernel outputs match the eager dequantized reference to
fp32 dot-reassociation noise.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import FP8_DTYPE, FP8_MAX, QBLOCK

_EPS = 1e-12


class ActQuant(NamedTuple):
    """Per-1x128-group quantized activation."""

    q: jax.Array      # [M, K] e4m3
    s_t: jax.Array    # [K/QBLOCK, M] fp32 (transposed; see module docstring)


class WeightQuant(NamedTuple):
    """Per-128x128-block quantized weight (optionally expert-grouped)."""

    q: jax.Array      # [K, N] or [E, K, N] e4m3
    s: jax.Array      # [K/QBLOCK, N/QBLOCK] or [E, K/QBLOCK, N/QBLOCK] fp32


def _to_fp8(x: jax.Array) -> jax.Array:
    return jnp.clip(x, -FP8_MAX, FP8_MAX).astype(FP8_DTYPE)


def quantize_act(x: jax.Array) -> ActQuant:
    """Quantize ``x[M, K]`` per 1x128 group. K must be a QBLOCK multiple."""
    M, K = x.shape
    if K % QBLOCK != 0:
        raise ValueError(f"K={K} must be a multiple of {QBLOCK}")
    xf = x.astype(jnp.float32).reshape(M, K // QBLOCK, QBLOCK)
    amax = jnp.max(jnp.abs(xf), axis=-1)                       # [M, K/128]
    s = jnp.maximum(amax, _EPS) / FP8_MAX                       # [M, K/128]
    q = _to_fp8(xf / s[..., None]).reshape(M, K)
    return ActQuant(q=q, s_t=s.T)                               # s_t: [K/128, M]


def dequantize_act(aq: ActQuant) -> jax.Array:
    """fp32 reference dequantization."""
    K2, M = aq.s_t.shape
    qf = aq.q.astype(jnp.float32).reshape(M, K2, QBLOCK)
    return (qf * aq.s_t.T[..., None]).reshape(M, K2 * QBLOCK)


def quantize_weight(w: jax.Array) -> WeightQuant:
    """Quantize ``w[..., K, N]`` per 128x128 block (leading dims = experts)."""
    *lead, K, N = w.shape
    if K % QBLOCK != 0 or N % QBLOCK != 0:
        raise ValueError(f"(K={K}, N={N}) must be multiples of {QBLOCK}")
    wf = w.astype(jnp.float32).reshape(
        *lead, K // QBLOCK, QBLOCK, N // QBLOCK, QBLOCK)
    amax = jnp.max(jnp.abs(wf), axis=(-3, -1))                  # [..., K/128, N/128]
    s = jnp.maximum(amax, _EPS) / FP8_MAX
    q = _to_fp8(wf / s[..., :, None, :, None]).reshape(*lead, K, N)
    return WeightQuant(q=q, s=s)


def dequantize_weight(wq: WeightQuant) -> jax.Array:
    """fp32 reference dequantization."""
    *lead, K, N = wq.q.shape
    qf = wq.q.astype(jnp.float32).reshape(
        *lead, K // QBLOCK, QBLOCK, N // QBLOCK, QBLOCK)
    return (qf * wq.s[..., :, None, :, None]).reshape(*lead, K, N)


# ---------------------------------------------------------------------------
# Hybrid KV quantization (paper §2.3.4)
# ---------------------------------------------------------------------------
#
# Compressed KV entries store the rope dims (last 64) in bf16 and the
# remaining "nope" dims in fp8 with one fp32 scale per entry. The nope/rope
# *split* storage (rather than concat-then-split) is a kernel constraint:
# inside Pallas the lane axis can't be concatenated at a non-128 boundary,
# so the attention kernel runs a split dot (nope fp8 dot * row scale +
# rope bf16 dot) and never materializes the concatenated entry.


class KVQuant(NamedTuple):
    nope: jax.Array     # [..., S, c - rope] e4m3
    rope: jax.Array     # [..., S, rope]     bf16
    scale: jax.Array    # [..., S, 1]        fp32 (per entry, nope dims)


def quantize_kv(k: jax.Array, rope_dim: int) -> KVQuant:
    """Split ``k[..., S, c]`` into hybrid fp8(nope)+bf16(rope) storage."""
    nope = k[..., : k.shape[-1] - rope_dim].astype(jnp.float32)
    rope = k[..., k.shape[-1] - rope_dim :]
    amax = jnp.max(jnp.abs(nope), axis=-1, keepdims=True)       # [..., S, 1]
    s = jnp.maximum(amax, _EPS) / FP8_MAX
    return KVQuant(nope=_to_fp8(nope / s), rope=rope.astype(jnp.bfloat16), scale=s)


def dequantize_kv(kv: KVQuant) -> jax.Array:
    """fp32 reference reconstruction ``[..., S, c]``."""
    nope = kv.nope.astype(jnp.float32) * kv.scale
    return jnp.concatenate([nope, kv.rope.astype(jnp.float32)], axis=-1)
