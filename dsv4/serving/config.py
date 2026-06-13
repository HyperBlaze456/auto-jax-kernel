"""Serving-side configuration for the DSv4 full-model forward pass.

This module is the single place where the *hardware plan* lives: model
shapes (from the V4 paper §4.2.1), the precision plan (paper §2.3.4, §3.4),
and the parallelism plan (paper §3.1). Kernels read resolved tile sizes
from here the same way ``kernel_v1`` reads ``KernelConfig``.

Precision plan (paper-faithful)
-------------------------------

========================  =========================================  =====
tensor                    storage                                    compute
==========================================================================
MoE expert weights        FP8 E4M3 + f32 scale per 128x128 block     fp8 MXU (or
                          (lossless dequant from FP4 master)         bf16-upcast),
                                                                     fp32 accum
MoE activations           FP8 E4M3 + f32 scale per 1x128 group       same
dispatch payload          FP8 + scales (a2a)                         --
combine payload           BF16 (a2a)                                 --
KV cache (nope dims)      FP8 E4M3 + f32 scale per row               folded into
                                                                     logits / PV
KV cache (rope dims)      BF16                                       bf16 dot
attention accumulators    --                                         fp32 (m, l, acc)
router / sinkhorn / lse   --                                         fp32
==========================================================================

The fp8 compute path is *bit-exact* between ``compute_upcast=True``
(e4m3 -> bf16 upcast before the MXU dot; every e4m3 value is exactly
representable in bf16) and native fp8 MXU dots — only throughput differs.
That is what lets one kernel source serve v5e (no fp8 MXU) and v6e+
(native fp8) without a numerics fork.

ICI/DCN plan
------------

Mesh axes: ``("data", "expert")``. Attention + dense layers are replicated
over "expert" (sharded over "data" only); MoE expert weights are sharded
over "expert". Dispatch/combine are all-to-alls over "expert", split into
``n_waves`` independent waves so XLA's latency-hiding scheduler can overlap
wave w's a2a with wave w-1's grouped GEMM (paper §3.1's fine-grained EP,
expressed at the dependency-graph level rather than as one mega-kernel).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ..eager import CSAConfig, HCAConfig
from ..kernel_config import TpuSpec, detect_tpu


# Quantization block sizes. These mirror the DeepGEMM / DSv3 convention and
# the paper's FP8 framework (128x128 weight blocks, 1x128 activation groups).
QBLOCK: int = 128
FP8_MAX: float = 448.0          # e4m3fn finite max
FP8_DTYPE = jnp.float8_e4m3fn


@dataclass(frozen=True)
class MoEConfig:
    """DeepSeekMoE shape + routing knobs (paper §2.1, §4.2.1)."""

    n_routed: int               # routed experts (256 Flash / 384 Pro)
    d_expert: int               # expert FFN hidden (2048 Flash / 3072 Pro)
    topk: int = 6               # activated routed experts per token
    n_shared: int = 1           # shared experts (dense, always-on)
    n_hash_layers: int = 3      # first N MoE layers use hash routing
    # Routing affinity is Sqrt(Softplus(logit)) in V4 (changed from V3's
    # sigmoid). Aux-loss-free bias is added for *selection only*; gate
    # weights come from the unbiased affinities, normalized over the top-k.
    route_norm_topk: bool = True
    # Per-EP-shard receive capacity factor (x mean tokens/shard). Tokens
    # beyond capacity are dropped from dispatch (gate mass renormalized);
    # serving deployments size this so drops are ~never hit.
    capacity_factor: float = 1.5


@dataclass(frozen=True)
class PrecisionConfig:
    """What gets stored/computed in what dtype. See module docstring."""

    fp8: bool = True                 # False = pure-bf16 path (debug / oracle)
    compute_upcast: bool = True      # upcast fp8->bf16 before MXU dot.
    #   True : exact same numerics, works on every TPU generation.
    #   False: native fp8 MXU dot (v6e+ / v7); same math, 2x MXU rate.
    kv_fp8: bool = True              # hybrid KV storage (fp8 nope + bf16 rope)
    dispatch_fp8: bool = True        # quantize EP dispatch payload


@dataclass(frozen=True)
class ParallelConfig:
    """Mesh + overlap plan."""

    ep_size: int = 1            # devices along the "expert" mesh axis
    dp_size: int = 1            # devices along the "data" mesh axis
    n_waves: int = 4            # expert waves for dispatch/compute overlap
    # ICI is the assumed transport for the EP axis. If the expert axis
    # spans DCN (multi-slice), capacity_factor and n_waves should both
    # rise: DCN latency >> ICI, so deeper pipelining is required to hide it.
    ep_over_dcn: bool = False


@dataclass(frozen=True)
class ServingTiles:
    """Resolved per-kernel tile sizes (one ``TpuSpec`` -> one of these)."""

    # Grouped FP8 GEMM: rows per m-tile. N is not tiled (full-N expert
    # tiles: weights stream exactly once per m-tile) and the K grid step
    # is locked to QBLOCK=128 — one quant block per step is what makes
    # every scale lookup a pure BlockSpec index_map (see gemm_fp8).
    gemm_tm: int = 128
    # Gather attention: top-k rows fetched per DMA wave.
    attn_chunk: int = 128
    # mHC fused kernels: tokens per program.
    mhc_bn: int = 128
    # Run all pallas_calls in interpret mode (CPU dev / tests).
    interpret: bool = False


@dataclass(frozen=True)
class ModelConfig:
    """Full DSv4 stack: layer schedule + sub-configs."""

    d: int
    n_layers: int
    vocab: int
    csa: CSAConfig
    hca: HCAConfig
    moe: MoEConfig
    # Grouped output projection (paper §2.3.1): n_h heads split into g
    # groups; each c*(n_h/g) group -> d_g, then g*d_g -> d.
    g: int
    d_g: int
    # mHC (paper §2.2)
    hc: int = 4
    sinkhorn_iters: int = 20
    # Layer schedule: first `n_swa_only` layers use the SWA-only/HCA intro
    # pattern; afterwards CSA/HCA interleave starting with CSA.
    n_swa_only: int = 2
    intro_kind: str = "swa"      # "swa" (Flash) | "hca" (Pro) for the intro layers
    # CSA selection granularity (HARDWARE_NOTES §13). 0 = row-exact top-k
    # over every compressed entry (the default contract). P > 0 = coarse-
    # to-fine: the indexer scans only per-page summaries (n_blk/P of them)
    # and selects topk/P pages of P consecutive entries, gathered by the
    # paged kernel — cutting both the dominant long-context scan bytes and
    # the gather descriptor count by ~P. Requires csa.topk % P == 0 and
    # P * csa.m <= csa.n_win (the in-progress page is then always covered
    # by the raw SWA window, so completed-pages-only selection loses no
    # reachable context).
    csa_pages: int = 0
    # Exact coarse-to-fine (HARDWARE_NOTES §13.7), requires csa_pages > 0.
    # The page summaries become coordinatewise max/min *envelopes* whose
    # scores are sound per-page upper bounds; decode rescans the top
    # `csa_rescan` pages (0 = auto: 2 * topk // P) at row resolution and
    # takes the row-exact f32 top-k over the candidates. Selection equals
    # the csa_pages=0 contract whenever the candidate set covers the true
    # top-k (the certificate the tests assert); the scan still reads only
    # n_blk/P-sized summaries — 2 caches + a topk-sized rescan instead of
    # 1 cache, ~2x the mean-summary bytes, still ~P/2x below a full scan.
    csa_pages_exact: bool = False
    csa_rescan: int = 0


def _flash() -> ModelConfig:
    return ModelConfig(
        d=4096, n_layers=43, vocab=129280,
        csa=CSAConfig(d=4096, c=512, n_h=64, m=4, topk=512, n_win=128,
                      d_c=1024, n_I_h=64, c_I=128),
        hca=HCAConfig(d=4096, c=512, n_h=64, m_prime=128, n_win=128, d_c=1024),
        moe=MoEConfig(n_routed=256, d_expert=2048),
        g=8, d_g=1024,
        intro_kind="swa",
    )


def _pro() -> ModelConfig:
    return ModelConfig(
        d=7168, n_layers=61, vocab=129280,
        csa=CSAConfig(d=7168, c=512, n_h=128, m=4, topk=1024, n_win=128,
                      d_c=1536, n_I_h=64, c_I=128),
        hca=HCAConfig(d=7168, c=512, n_h=128, m_prime=128, n_win=128, d_c=1536),
        moe=MoEConfig(n_routed=384, d_expert=3072),
        g=16, d_g=1024,
        intro_kind="hca",
    )


def _small() -> ModelConfig:
    """Tiny CPU/dev preset. Same algorithmic surface; dims sized so every
    quantization block / tile constraint is exercised (>= one full 128 lane
    group on the GEMM paths) while staying fast in interpret mode."""
    return ModelConfig(
        d=256, n_layers=4, vocab=512,
        csa=CSAConfig(d=256, c=128, n_h=4, m=4, topk=8, n_win=16,
                      d_c=64, n_I_h=4, c_I=32),
        hca=HCAConfig(d=256, c=128, n_h=4, m_prime=16, n_win=16, d_c=64),
        moe=MoEConfig(n_routed=8, d_expert=256, topk=2, n_hash_layers=1),
        g=2, d_g=128,
        n_swa_only=1,
        intro_kind="swa",
    )


DSV4_FLASH = _flash()
DSV4_PRO = _pro()
SMALL_MODEL = _small()


def tiles_for(spec: TpuSpec | None = None, *, interpret: bool | None = None) -> ServingTiles:
    """Resolve tile sizes for a TPU generation (auto-detected by default).

    The defaults are sized against the *smallest* current VMEM budget
    (32 MiB on v4/v6e): the grouped-GEMM working set at tm=128 over the
    Pro expert shapes is ~12 MiB incl. double buffering, the gather-attn
    working set ~2 MiB. Larger-VMEM parts (v5p/v6p, 64 MiB+) can raise
    ``gemm_tm`` to 256/512 for prefill-heavy serving.
    """
    if spec is None:
        spec = detect_tpu()
    if spec is None:
        # Non-TPU host: interpret mode, tiles small enough for fast tests.
        return ServingTiles(gemm_tm=128, attn_chunk=8, mhc_bn=8,
                            interpret=True if interpret is None else interpret)
    big_vmem = spec.vmem_bytes >= 64 * 1024 * 1024
    return ServingTiles(
        gemm_tm=256 if big_vmem else 128,
        attn_chunk=128,
        mhc_bn=128,
        interpret=False if interpret is None else interpret,
    )
