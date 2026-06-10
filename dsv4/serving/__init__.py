"""DSv4 TPU serving suite: full-model forward with HBM/ICI-minimal kernels.

Modules:
  config         — model/precision/parallelism plans + tile resolution
  quant          — fp8 block-scale quantization (acts, weights, hybrid KV)
  gemm_fp8       — grouped block-scaled FP8 GEMM + fused SwiGLU-quant (Pallas)
  attention      — fused-gather sparse MQA with sink, prefill+decode (Pallas)
  moe            — routing, EP dispatch/combine, wave overlap
  mhc            — fused hyper-connection residual ops (Pallas)
  model          — full-stack prefill / decode_step
  attention_paged— page-aligned selection + paged gather kernel (one DMA
                   per P-entry page; descriptor-bound → bandwidth-bound)
  moe_megakernel — fused dispatch/expert-GEMM/combine remote-DMA wave
                   pipeline (one Pallas kernel per shard; MegaMoE on TPU)

Training (backward) counterparts:
  gemm_fp8_diff  — diffable grouped expert FFN (fp8 fwd, bf16 bwd, fp8
                   residuals + SwiGLU recompute; dgrad/swiglu-bwd Pallas,
                   wgrad via megablox tgmm)
  attention_train— bf16-KV gather attention fwd(o, lse) + re-gather bwd
                   with deterministic scatter-add dK reduction
  mhc_diff       — closed-form fused Pallas backwards for the mHC ops
  moe_diff       — jax.grad-able MoE layer (custom_vjp only on the FFN)
  train_step     — full-model (loss, grads) step with per-layer remat;
                   to_serving_params closes the QAT train→serve loop

See HARDWARE_NOTES.md for the byte-accounting design rationale (§8 for
the backward pass).
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
