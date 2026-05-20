"""Hardware-spec-driven configuration for the v1 Pallas sparse-attn kernel.

The point of this module is to make ``dsv4/kernel_v1.py`` *generation
portable*: the same kernel source serves v4, v5e, v5p, v6e, v6p, and any
future v7+ part. The kernel reads block sizes and lane/sublane constants
from a ``KernelConfig``; adding a new generation only requires registering a
new ``TpuSpec`` here — no edits to ``kernel_v1.py``.

How the pieces fit together
---------------------------

``TpuSpec`` — hardware shape: VMEM per tensorcore, megacore-or-not, lane &
sublane widths. One per generation. Public-doc numbers; override via
``register_tpu`` if a particular host disagrees.

``KernelConfig`` — what the kernel actually consumes: the resolved
``(BQ, BS)`` block sizes plus the lane width (so the kernel's alignment
check doesn't bake in 128). Hashable & frozen so it can key the closure
cache in ``kernel_v1``.

``config_for(spec, n_h=, c=)`` — chooses ``(BQ, BS)`` for a problem on a
given ``TpuSpec``. Prefers larger ``BS`` (better QK/PV arithmetic
intensity), then larger ``BQ``, subject to a VMEM budget that mirrors the
kernel's actual tile footprint.

``detect_tpu()`` — best-effort match of the local device kind to a
registered ``TpuSpec``. Falls back to ``None`` so callers can supply an
explicit spec on non-TPU backends or unrecognized devices.

Usage
-----

Auto-detect (most callers)::

    from dsv4 import kernel_v1
    out = kernel_v1.sparse_attn_kernel_v1(q, K_comp, topk_idxs, K_swa, sink)

Pin a generation, let the resolver pick block sizes::

    from dsv4.kernel_config import config_for
    cfg = config_for("v5p", n_h=128, c=512)
    out = kernel_v1.sparse_attn_kernel_v1(q, ..., config=cfg)

Fully custom (e.g. autotuned externally)::

    from dsv4.kernel_config import KernelConfig
    cfg = KernelConfig(bq=16, bs=256)
    out = kernel_v1.sparse_attn_kernel_v1(q, ..., config=cfg)

Register a new generation without touching the kernel::

    from dsv4.kernel_config import register_tpu, TpuSpec, MiB
    register_tpu(TpuSpec("v7p", vmem_bytes=128 * MiB, num_cores=2))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import jax


MiB: Final[int] = 1024 * 1024


# ---------------------------------------------------------------------------
# Spec types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TpuSpec:
    """Per-generation hardware knobs the resolver consumes.

    Values are *per tensorcore*. Megacore parts (v5p, v6p) expose two cores
    per chip; the compiler splits the kernel's ``"parallel"`` grid axes
    across them, but each program still has to fit in one core's VMEM.
    """

    name: str
    vmem_bytes: int          # VMEM per tensorcore
    num_cores: int = 1       # 2 for megacore parts; 1 otherwise
    lane_size: int = 128     # VPU lane width (BS must be a multiple of this)
    sublane_size: int = 8    # second-minor alignment requirement


@dataclass(frozen=True)
class KernelConfig:
    """Block sizes + lane width resolved for one (TpuSpec, problem) pair.

    Frozen so it's hashable — ``kernel_v1`` / ``kernel_v2`` key an
    ``lru_cache`` of closure-built kernels on it, which keeps the JIT
    cache hit-rate intact across calls with the same config.
    """

    bq: int                  # queries per Pallas program
    bs: int                  # KV-seq positions per inner accumulation step
    lane_size: int = 128
    use_megacore: bool = False
    # Sinkhorn tile size (queries per Pallas program). Defaults to lane_size
    # so the per-token reductions are 8×128-aligned without padding.
    bn_sinkhorn: int = 128
    # Run pallas_call in interpret mode (CPU simulation). Useful for local
    # numerical validation against the reference before deploying to TPU;
    # ignored on TPU backends where it would just slow things down.
    interpret: bool = False


# ---------------------------------------------------------------------------
# Registry of known TPU generations
# ---------------------------------------------------------------------------

# Public-doc figures. If your host disagrees (e.g. reserved scratch reduces
# usable VMEM), override via ``register_tpu`` at process start.
TPU_SPECS: dict[str, TpuSpec] = {
    "v4":  TpuSpec("v4",  vmem_bytes=32 * MiB, num_cores=1),
    "v5e": TpuSpec("v5e", vmem_bytes=48 * MiB, num_cores=1),
    "v5p": TpuSpec("v5p", vmem_bytes=64 * MiB, num_cores=2),
    "v6e": TpuSpec("v6e", vmem_bytes=32 * MiB, num_cores=1),
    "v6p": TpuSpec("v6p", vmem_bytes=64 * MiB, num_cores=2),
    # Dev fallback for CPU / unknown backends. ``lane_size=1`` disables
    # the VPU-alignment requirement so tiny dev shapes (e.g. SMALL_CSA
    # with c=64) trace through pallas_call(interpret=True) without
    # tripping the multiple-of-128 check. VMEM is generous because
    # interpret mode doesn't actually allocate VMEM.
    "cpu": TpuSpec("cpu", vmem_bytes=128 * MiB, num_cores=1, lane_size=1),
}


def register_tpu(spec: TpuSpec) -> None:
    """Add or override a ``TpuSpec``. Use for v7+ or host-specific tuning."""
    TPU_SPECS[spec.name] = spec
    # Auto-extend detection if this name maps to a recognizable substring.
    # Users can also extend _KIND_PATTERNS directly for nonstandard kinds.


# ---------------------------------------------------------------------------
# Resolution: TpuSpec + problem dims → KernelConfig
# ---------------------------------------------------------------------------


def _tile_bytes(*, bq: int, bs: int, n_h: int, c: int, dtype_bytes: int) -> int:
    """Per-program VMEM footprint, mirroring the tiles in ``kernel_v1``.

    Tiles tracked (must stay in sync with ``_flash_attn_with_sink_pallas``):

      - q + o            : 2 · bq · n_h · c · dtype
      - k (== v in V4)   : bq · bs · c · dtype
      - mask (bool)      : bq · bs
      - m, l, acc (fp32) : 3 · bq · n_h · c · 4

    Doesn't model double-buffered pipeline scratch the compiler may add for
    the inner k-loop; ``safety_factor`` in ``config_for`` reserves headroom.
    """
    return (
        2 * bq * n_h * c * dtype_bytes
        + bq * bs * c * dtype_bytes
        + bq * bs
        + 3 * bq * n_h * c * 4
    )


def config_for(
    spec: TpuSpec | str,
    *,
    n_h: int,
    c: int,
    dtype_bytes: int = 2,
    safety_factor: float = 0.6,
) -> KernelConfig:
    """Pick ``(BQ, BS)`` for ``spec`` given problem dims.

    Strategy: largest ``BS`` that fits the budget (better arithmetic
    intensity on QK^T and PV), then the largest ``BQ`` at that ``BS``.

    ``safety_factor`` reserves VMEM for pipeline scratch the
    simple per-program estimate doesn't model — drop it if you're hitting
    OOC on a generation that's leaner about that.

    Raises ``ValueError`` if nothing fits; caller can fall back to a
    handcrafted ``KernelConfig`` or relax the safety factor.
    """
    if isinstance(spec, str):
        spec = TPU_SPECS[spec]

    if c % spec.lane_size != 0:
        raise ValueError(
            f"head dim c={c} must be a multiple of lane_size={spec.lane_size} "
            f"for {spec.name}"
        )

    budget = int(spec.vmem_bytes * safety_factor)

    bs_candidates = (1024, 512, 256, spec.lane_size)
    bq_candidates = (32, 16, 8, 4, 2, 1)

    for bs in bs_candidates:
        for bq in bq_candidates:
            fits = _tile_bytes(
                bq=bq, bs=bs, n_h=n_h, c=c, dtype_bytes=dtype_bytes
            ) <= budget
            if fits:
                return KernelConfig(
                    bq=bq,
                    bs=bs,
                    lane_size=spec.lane_size,
                    use_megacore=spec.num_cores > 1,
                )

    raise ValueError(
        f"No (BQ, BS) fits the {spec.name} VMEM budget "
        f"(budget={budget} B at safety_factor={safety_factor}, "
        f"n_h={n_h}, c={c}, dtype_bytes={dtype_bytes}). "
        f"Pass an explicit KernelConfig(...) or relax safety_factor."
    )


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

# device_kind substrings → TPU_SPECS key. Order matters: more specific
# patterns must come first ("v5 lite" before "v5"), so e+lite parts don't
# get classified as their p siblings.
_KIND_PATTERNS: list[tuple[str, str]] = [
    ("v6 lite", "v6e"),
    ("v6e",     "v6e"),
    ("v6p",     "v6p"),
    ("v6",      "v6p"),
    ("v5 lite", "v5e"),
    ("v5e",     "v5e"),
    ("v5p",     "v5p"),
    ("v5",      "v5p"),
    ("v4",      "v4"),
]


def detect_tpu() -> TpuSpec | None:
    """Best-effort lookup of the local TPU's spec.

    Returns ``None`` on non-TPU backends or when ``device_kind`` doesn't
    match any registered pattern. Callers should fall back to an explicit
    spec rather than trusting silent defaults.
    """
    try:
        devices = jax.devices()
    except Exception:
        return None
    if not devices or devices[0].platform != "tpu":
        return None

    kind = devices[0].device_kind.lower()
    for needle, key in _KIND_PATTERNS:
        if needle in kind:
            return TPU_SPECS.get(key)
    return None


def default_config(*, n_h: int, c: int, dtype_bytes: int = 2) -> KernelConfig:
    """Auto-detect a TPU spec and resolve a ``KernelConfig`` for it.

    Falls back to the v5p preset on non-TPU backends and flips
    ``interpret=True`` so ``pl.pallas_call`` runs in CPU simulation —
    that's enough for kernel_v2's correctness tests to execute locally.
    """
    spec = detect_tpu()
    if spec is None:
        # On a non-TPU backend, route through the CPU dev spec (lane_size=1
        # so tiny dev shapes don't fail alignment) and force interpret mode
        # so pallas_call simulates instead of trying to compile to TPU.
        cfg = config_for(TPU_SPECS["cpu"], n_h=n_h, c=c, dtype_bytes=dtype_bytes)
        return KernelConfig(
            bq=cfg.bq, bs=cfg.bs, lane_size=cfg.lane_size,
            use_megacore=cfg.use_megacore, bn_sinkhorn=cfg.bn_sinkhorn,
            interpret=True,
        )
    return config_for(spec, n_h=n_h, c=c, dtype_bytes=dtype_bytes)
