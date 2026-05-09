"""Benchmark + correctness harness for the DSv4 attention kernels.

Usage:
    python bench.py                                # dev preset, single shape
    python bench.py --preset dsv4_flash --seq 16384
    python bench.py --preset dsv4_pro --seq 65536 --bwd
    python bench.py --sweep                        # all presets × all seq lens
    JAX_PLATFORMS=tpu python bench.py --profile /tmp/dsv4_trace
    XLA_FLAGS=--xla_dump_to=/tmp/hlo python bench.py

Output: a single `---` block with kernel name → reference. Mirrors the
upstream autoresearch summary format so program.md can grep it.

The agent edits dsv4/kernel.py and re-runs this script. Lower fwd_latency_ms
wins, gated on max_abs_diff < tolerance.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from dsv4 import kernel, reference as ref


# ---------------------------------------------------------------------------
# TPU peak FLOPs table (bf16). Override via --peak-tflops when on a chip
# this script doesn't know about.
# ---------------------------------------------------------------------------
TPU_PEAK_BF16_TFLOPS = {
    "TPU v4":  275.0,
    "TPU v5e": 197.0,
    "TPU v5p": 459.0,
    "TPU v6e": 918.0,
}


def detect_peak_tflops(override: float | None) -> tuple[float, str]:
    if override is not None:
        return override, f"override={override}"
    devs = jax.devices()
    if not devs:
        return 1.0, "no-device"
    name = getattr(devs[0], "device_kind", repr(devs[0]))
    for k, v in TPU_PEAK_BF16_TFLOPS.items():
        if k.lower() in name.lower():
            return v, name
    # Non-TPU: report a placeholder so MFU is still computed but obviously not authoritative.
    return 1.0, name


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
    "small_csa":      ("csa", ref.SMALL_CSA),
    "small_hca":      ("hca", ref.SMALL_HCA),
    "dsv4_flash_csa": ("csa", ref.DSV4_FLASH_CSA),
    "dsv4_flash_hca": ("hca", ref.DSV4_FLASH_HCA),
    "dsv4_pro_csa":   ("csa", ref.DSV4_PRO_CSA),
    "dsv4_pro_hca":   ("hca", ref.DSV4_PRO_HCA),
}


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    median_ms: float
    p10_ms: float
    p90_ms: float
    iters: int


def _wait(x):
    """Block until a pytree of arrays is ready."""
    jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x)


def time_fn(fn: Callable, *args, warmup: int = 3, iters: int = 10) -> TimingResult:
    """Block-until-ready timing. Treats fn as already JIT'd."""
    # Warmup (also triggers compilation)
    for _ in range(warmup):
        out = fn(*args)
        _wait(out)

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        _wait(out)
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    return TimingResult(
        median_ms=statistics.median(samples),
        p10_ms=samples[max(0, int(0.1 * len(samples)) - 1)],
        p90_ms=samples[min(len(samples) - 1, int(0.9 * len(samples)))],
        iters=iters,
    )


# ---------------------------------------------------------------------------
# Per-preset bench
# ---------------------------------------------------------------------------

def bench_csa(cfg: ref.CSAConfig, B: int, n: int, do_bwd: bool, dtype, seed: int):
    key = jax.random.PRNGKey(seed)
    k_h, k_p = jax.random.split(key)
    H = jax.random.normal(k_h, (B, n, cfg.d), dtype=dtype)
    params = ref.init_csa_params(k_p, cfg, dtype=dtype)

    def fwd(H, params):
        return kernel.csa_forward_kernel(H, params, cfg)

    fwd_jit = jax.jit(fwd)
    ref_jit = jax.jit(lambda H, p: ref.csa_forward(H, p, cfg))

    # Correctness: run both, compare.
    y_kernel = fwd_jit(H, params); _wait(y_kernel)
    y_ref = ref_jit(H, params);    _wait(y_ref)
    max_abs_diff = float(jnp.max(jnp.abs(y_kernel - y_ref)))

    # Forward timing
    fwd_t = time_fn(fwd_jit, H, params)

    # Backward timing (sum-of-output as scalar loss)
    bwd_t = None
    if do_bwd:
        def loss_fn(H, params):
            return fwd(H, params).sum()
        grad_jit = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))
        bwd_t = time_fn(grad_jit, H, params)

    flops = ref.csa_flops(cfg, B, n)
    return y_kernel, max_abs_diff, fwd_t, bwd_t, flops


def bench_hca(cfg: ref.HCAConfig, B: int, n: int, do_bwd: bool, dtype, seed: int):
    key = jax.random.PRNGKey(seed)
    k_h, k_p = jax.random.split(key)
    H = jax.random.normal(k_h, (B, n, cfg.d), dtype=dtype)
    params = ref.init_hca_params(k_p, cfg, dtype=dtype)

    def fwd(H, params):
        return kernel.hca_forward_kernel(H, params, cfg)

    fwd_jit = jax.jit(fwd)
    ref_jit = jax.jit(lambda H, p: ref.hca_forward(H, p, cfg))

    y_kernel = fwd_jit(H, params); _wait(y_kernel)
    y_ref = ref_jit(H, params);    _wait(y_ref)
    max_abs_diff = float(jnp.max(jnp.abs(y_kernel - y_ref)))

    fwd_t = time_fn(fwd_jit, H, params)
    bwd_t = None
    if do_bwd:
        def loss_fn(H, params):
            return fwd(H, params).sum()
        grad_jit = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))
        bwd_t = time_fn(grad_jit, H, params)

    flops = ref.hca_flops(cfg, B, n)
    return y_kernel, max_abs_diff, fwd_t, bwd_t, flops


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(
    *, preset: str, B: int, n: int, max_abs_diff: float,
    fwd_t: TimingResult, bwd_t: TimingResult | None,
    flops: int, peak_tflops: float, peak_label: str,
    tol: float,
):
    fwd_tflops = (flops / 1e12) / (fwd_t.median_ms / 1000.0)
    fwd_mfu = 100.0 * fwd_tflops / peak_tflops if peak_tflops > 0 else 0.0
    status = "pass" if max_abs_diff < tol else "FAIL"

    print("---")
    print(f"preset:           {preset}")
    print(f"batch:            {B}")
    print(f"seq_len:          {n}")
    print(f"fwd_latency_ms:   {fwd_t.median_ms:.4f}    (p10={fwd_t.p10_ms:.4f}, p90={fwd_t.p90_ms:.4f}, iters={fwd_t.iters})")
    if bwd_t is not None:
        bwd_tflops = (3 * flops / 1e12) / (bwd_t.median_ms / 1000.0)  # 3× heuristic
        bwd_mfu = 100.0 * bwd_tflops / peak_tflops if peak_tflops > 0 else 0.0
        print(f"bwd_latency_ms:   {bwd_t.median_ms:.4f}")
        print(f"bwd_tflops:       {bwd_tflops:.2f}")
        print(f"bwd_mfu_percent:  {bwd_mfu:.2f}")
    print(f"fwd_tflops:       {fwd_tflops:.2f}")
    print(f"fwd_mfu_percent:  {fwd_mfu:.2f}")
    print(f"max_abs_diff:     {max_abs_diff:.3e}    (tol={tol:.0e})")
    print(f"peak_tflops:      {peak_tflops:.1f}    ({peak_label})")
    print(f"status:           {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one(preset: str, B: int, n: int, do_bwd: bool, dtype, peak_tflops: float, peak_label: str, tol: float, seed: int):
    kind, cfg = PRESETS[preset]
    if kind == "csa":
        if n % cfg.m != 0:
            raise SystemExit(f"seq_len {n} must be divisible by CSA m={cfg.m}")
        _, mad, fwd_t, bwd_t, flops = bench_csa(cfg, B, n, do_bwd, dtype, seed)
    else:
        if n % cfg.m_prime != 0:
            raise SystemExit(f"seq_len {n} must be divisible by HCA m'={cfg.m_prime}")
        _, mad, fwd_t, bwd_t, flops = bench_hca(cfg, B, n, do_bwd, dtype, seed)
    report(
        preset=preset, B=B, n=n,
        max_abs_diff=mad, fwd_t=fwd_t, bwd_t=bwd_t,
        flops=flops, peak_tflops=peak_tflops, peak_label=peak_label, tol=tol,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="small_csa", choices=list(PRESETS))
    p.add_argument("--seq", type=int, default=512, help="sequence length")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--bwd", action="store_true", help="also time backward pass")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--peak-tflops", type=float, default=None)
    p.add_argument("--tol", type=float, default=1e-2,
                   help="max-abs-diff tolerance vs reference (loose default for bf16)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile", default=None,
                   help="if set, dump a jax.profiler trace to this directory")
    p.add_argument("--sweep", action="store_true",
                   help="run all presets × a default seq-len ladder, ignore --preset/--seq")
    args = p.parse_args()

    dtype = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}[args.dtype]
    peak, peak_label = detect_peak_tflops(args.peak_tflops)

    if args.profile:
        os.makedirs(args.profile, exist_ok=True)
        jax.profiler.start_trace(args.profile)
        try:
            _do(args, dtype, peak, peak_label)
        finally:
            jax.profiler.stop_trace()
            print(f"\nprofile trace dumped to {args.profile}")
    else:
        _do(args, dtype, peak, peak_label)


def _do(args, dtype, peak, peak_label):
    if args.sweep:
        ladder_csa = [512, 4096, 16384, 65536]
        ladder_hca = [2048, 16384, 65536]
        for preset in PRESETS:
            ladder = ladder_csa if "csa" in preset else ladder_hca
            for n in ladder:
                if "small" not in preset and n >= 65536:
                    continue  # skip the huge full-size shapes from a default sweep
                try:
                    run_one(preset, args.batch, n, args.bwd, dtype, peak, peak_label, args.tol, args.seed)
                except Exception as e:
                    print(f"---\npreset:           {preset}\nseq_len:          {n}\nstatus:           ERROR\nerror:            {e}\n")
                gc.collect()
        return

    run_one(args.preset, args.batch, args.seq, args.bwd, dtype, peak, peak_label, args.tol, args.seed)


if __name__ == "__main__":
    main()
