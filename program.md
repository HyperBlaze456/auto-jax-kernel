# auto-jax-kernel

You are an autonomous kernel-development agent. Your job is to write the
fastest possible Pallas-TPU kernel for DeepSeek V4's hybrid attention
(Compressed Sparse Attention, Heavily Compressed Attention, and the short
sliding-window branch), gated on bit-close correctness against an eager-JAX
reference.

This is **not** model-training autoresearch. You are not optimising val_bpb.
You are optimising kernel latency.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`).
   The branch `kernel/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b kernel/<tag>` from the current
   working branch (probably `dsv4-kernel` or `master`).
3. **Read the in-scope files**:
   - `DeepSeek_V4.pdf` §2.3 — the spec (CSA, HCA, SWA, attention sink, RoPE).
   - `kernel_refs.md` — curated Pallas-TPU + sharding reference card. Read
     this **before** writing kernel code. Has the canonical FlashAttention
     forward, the scalar-prefetched block-sparse pattern (which is exactly
     what our top-k MQA needs), the four matmul-sharding cases, layout
     rules, and the footgun list.
   - `dsv4/reference.py` — the eager-JAX reference. **Do not modify.** It is
     the ground-truth oracle.
   - `dsv4/kernel.py` — the file you modify. Surface contract documented at
     the top of the file. Replace the eager bodies with `@pl.pallas_call`
     kernels.
   - `bench.py` — the harness. Don't modify unless it is genuinely broken;
     prefer to extend with new flags rather than change defaults.
4. **Verify the env**: `uv sync` and `uv run python bench.py --preset
   small_csa --seq 64`. You should see `status: pass` with `max_abs_diff:
   0.000e+00` (since the v1 kernel just dispatches to the reference).
5. **Initialize results.tsv**: header row only. The baseline is recorded
   after the first real run.
6. **Confirm and go.**

## Experimentation

Each experiment edits `dsv4/kernel.py` to make some part of the surface
faster, and is graded by `bench.py`.

**What you CAN do:**
- Modify `dsv4/kernel.py`: replace the body of `csa_forward_kernel`,
  `hca_forward_kernel`, `sparse_attn_kernel`, or `mhc_sinkhorn_kernel` with a
  Pallas-TPU implementation. Replace the autodiff backward with a hand-written
  custom_vjp where it pays off.
- Add new helpers in a `dsv4/kernel_*.py` file if a kernel grows large.
- Set `XLA_FLAGS` for HLO dumps when profiling.

**What you CANNOT do:**
- Modify `dsv4/reference.py`. It is the oracle.
- Modify the public surface of `dsv4/kernel.py` (the function names, signatures,
  return dtypes/shapes). The bench script depends on them.
- Loosen the correctness tolerance in `bench.py` to make a kernel "pass". If
  you need to relax tolerance for legitimate numeric reasons (e.g. bf16
  accumulation differences), justify it in the description column.
- Change the FLOP counters in `reference.py` to inflate MFU.
- Install new packages outside what's in `pyproject.toml`. Pallas ships with
  `jax`; you should not need anything else.

**Goal: lower `fwd_latency_ms` (primary), raise `fwd_mfu_percent` (secondary),
without violating `max_abs_diff < tol`.** Backward latency is a secondary
target — once you have a fast forward, write a custom backward.

**Simplicity criterion**: same as the upstream. Fewer lines of kernel that
hit a given speed beats more lines. A 1.05× speedup that doubles kernel size
is rarely worth it; a 1.5× speedup that halves kernel size always is.

**The first run**: baseline. Run `python bench.py --preset small_csa --seq
4096` (or whatever your hardware reasonably handles for CSA), and a separate
HCA run. Record both as `baseline` in results.tsv. The v1 kernel dispatches
to the reference, so this measures the eager-JAX path — the bar everything
must beat.

## Output format

`bench.py` finishes with a summary like:

```
---
preset:           dsv4_flash_csa
batch:            1
seq_len:          16384
fwd_latency_ms:   12.3450    (p10=12.10, p90=12.78, iters=10)
fwd_tflops:       42.10
fwd_mfu_percent:  19.30
max_abs_diff:     1.420e-04    (tol=1e-02)
peak_tflops:      459.0    (TPU v5p)
status:           pass
```

Extract:
```
grep -E "^(preset|seq_len|fwd_latency_ms|fwd_mfu_percent|max_abs_diff|status):" run.log
```

## Logging results

Log each experiment to `results.tsv` (tab-separated). Columns:

```
commit  preset  seq_len  fwd_latency_ms  fwd_mfu_percent  max_abs_diff  status  description
```

1. git commit hash (short, 7 chars)
2. preset name (e.g. `dsv4_flash_csa`)
3. sequence length
4. fwd_latency_ms (or `0.0` if crashed / failed correctness)
5. fwd_mfu_percent (or `0.0`)
6. max_abs_diff (or `nan`)
7. status: `keep`, `discard`, `crash`, `fail` (correctness fail)
8. short text description of what this experiment tried

Example:
```
commit  preset           seq_len  fwd_latency_ms  fwd_mfu_percent  max_abs_diff  status   description
a1b2c3d dsv4_flash_csa   16384    180.4           1.5              0.0           keep     baseline (eager reference)
b2c3d4e dsv4_flash_csa   16384    72.1            3.8              1.2e-4        keep     pallas sparse_attn, block=64, 4-way wave
c3d4e5f dsv4_flash_csa   16384    0.0             0.0              nan           crash    block=128 OOM in VMEM
d4e5f6g dsv4_flash_csa   16384    71.9            3.8              5.1e-1        fail     dropped K-dim accumulator to bf16, lost precision
```

Do **not** commit `results.tsv` itself — leave it untracked.

## The experiment loop

You operate on a dedicated branch (e.g. `kernel/mar5`).

LOOP FOREVER:

1. Look at the git state — current branch and commit.
2. Tune `dsv4/kernel.py` with one experimental idea. Examples:
   - "Replace `sparse_attn_kernel` body with a Pallas kernel that blocks
     (n, k+n_win) at 64×64 and accumulates QK in fp32."
   - "Add a custom backward that reuses the saved logits."
   - "Fuse the lightning-indexer matmul + ReLU + head-mix into one kernel."
   - "Pull `csa_compress` into Pallas to get the softmax into VMEM."
3. `git commit -am "<short description>"`
4. Run `uv run python bench.py --preset <preset> --seq <N> > run.log 2>&1`.
   Redirect everything; do not let the output flood your context.
5. Read out the results: `grep -E "^(fwd_latency_ms|fwd_mfu_percent|max_abs_diff|status):" run.log`.
6. If grep is empty or `status:` is `FAIL`, run `tail -n 80 run.log` and
   either fix or give up after a few tries.
7. Record results in results.tsv.
8. If `fwd_latency_ms` improved AND `status: pass`, advance the branch.
9. Otherwise `git reset --hard HEAD~1` back to where you started.

**Hardware**: Pallas-TPU only runs on TPU. If you're on CPU/GPU and trying
to validate algorithm shape, run with the small preset (`--preset
small_csa`); the same code falls back to a reference path that runs
everywhere. Real benchmarks need a TPU.

**Profiling**: `python bench.py --profile /tmp/trace ...` dumps a jax.profiler
trace you can open in TensorBoard / Perfetto. For HLO inspection, set
`XLA_FLAGS="--xla_dump_to=/tmp/hlo --xla_dump_hlo_as_text"` before invoking.
Use `jax.named_scope("...")` liberally so the trace is legible.

**Numerics**: bf16 matmul + fp32 accumulate is the default. If you need to
deviate, document why in the description column. The reference is fp32 so
a bf16 kernel will have non-zero `max_abs_diff` — that's expected.

**Timeout**: each experiment should finish in well under 60 seconds wall.
If a run exceeds 5 minutes, kill it and treat as a failure.

**Crashes**: a Pallas kernel that fails to compile is a crash, not a fail.
Read the error, decide if it's a typo or a fundamental shape issue, fix or
skip. Common Pallas-TPU traps: VMEM overflow at large block sizes, missing
`block_shape` for the contraction dim, `lax.dot_general` precision flags.

**NEVER STOP**: once the experiment loop has begun, do NOT pause to ask the
human if you should continue. The human might be asleep. Run until manually
stopped. If you genuinely run out of ideas, re-read §2.3 of the paper, look
again at the reference TileLang `sparse_attn_kernel` (linked from the V4
HuggingFace repo), and consider:
- Different tile shapes (32×64, 64×64, 128×64).
- Pipelining the topk gather and the GEMM.
- Two-stage warp-specialised approach (the TileLang reference uses
  `T.Pipelined(num_blocks, num_stages=2)`).
- Custom backward that fuses the dQ + dK accumulation following the paper's
  per-SM accumulation buffer scheme (§3.3 — "Attention Backward").

## What good looks like

- The eager-reference `csa_forward` baseline is the floor. Any Pallas kernel
  that beats it on TPU is a win.
- TPU v5p peak bf16 is ~459 TF/s. Mature attention kernels on TPU hit
  35-50% MFU on long-context. Treat 25% MFU on `dsv4_flash_csa` at seq 64K
  as a meaningful milestone for a hand-rolled kernel.
- Backward should be no worse than ~3× forward. The default autodiff path is
  often 5-10×; replacing it with a fused custom_vjp is usually a clear win.

## Where the code lives

```
DeepSeek_V4.pdf      — spec (read-only)
kernel_refs.md       — Pallas-TPU + sharding reference card (read first)
dsv4/reference.py    — oracle (read-only)
dsv4/kernel.py       — agent edits this
bench.py             — harness (extend, don't change defaults)
program.md           — this file (don't edit)
pyproject.toml       — deps (don't edit unless adding a kernel-only tool)
```

`prepare.py`, `train.py`, `analysis.ipynb`, and `progress.png` are upstream
artifacts from karpathy/autoresearch; they are not used by this kernel work
and can be ignored (or deleted by the human).
