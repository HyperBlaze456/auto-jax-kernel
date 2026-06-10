"""DSv4 TPU serving suite: full-model forward with HBM/ICI-minimal kernels.

Modules:
  config     — model/precision/parallelism plans + tile resolution
  quant      — fp8 block-scale quantization (acts, weights, hybrid KV)
  gemm_fp8   — grouped block-scaled FP8 GEMM + fused SwiGLU-quant (Pallas)
  attention  — fused-gather sparse MQA with sink, prefill+decode (Pallas)
  moe        — routing, EP dispatch/combine, wave overlap
  mhc        — fused hyper-connection residual ops (Pallas)
  model      — full-stack prefill / decode_step

See HARDWARE_NOTES.md for the byte-accounting design rationale.
"""

from .config import (
    DSV4_FLASH,
    DSV4_PRO,
    SMALL_MODEL,
    ModelConfig,
    MoEConfig,
    ParallelConfig,
    PrecisionConfig,
    ServingTiles,
    tiles_for,
)

__all__ = [
    "DSV4_FLASH", "DSV4_PRO", "SMALL_MODEL",
    "ModelConfig", "MoEConfig", "ParallelConfig", "PrecisionConfig",
    "ServingTiles", "tiles_for",
]
