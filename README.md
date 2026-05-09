# auto-jax-kernel

An autoresearch-style harness for building a fast Pallas-TPU kernel for
DeepSeek V4's hybrid attention — Compressed Sparse Attention (CSA), Heavily
Compressed Attention (HCA), and the short sliding-window branch.

This fork is **not** a model trainer. The autoresearch loop structure
([upstream: karpathy/autoresearch](https://github.com/karpathy/autoresearch))
is reused, but the target the agent optimises is kernel latency, not
val_bpb. JAX/TPU is the host because XProf, HLO dumps, and Pallas give
better-grained signal for low-level kernel work than torch.profiler.

> Spec: `DeepSeek_V4.pdf`, §2.3 (CSA / HCA / SWA). Reference TileLang kernel:
> https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py

## What's here

```
DeepSeek_V4.pdf      spec (CSA, HCA, SWA, mHC Sinkhorn)
kernel_refs.md       Pallas-TPU + sharding reference card for the agent
dsv4/reference.py    eager-JAX oracle (read-only) — ground truth for correctness
dsv4/kernel.py       Pallas-TPU surface (agent edits this)
bench.py             latency + MFU + correctness harness
program.md           agent loop instructions
```

The kernel surface (must stay stable):

- `csa_forward_kernel(H, params, cfg)` — full CSA: token-compressor →
  lightning indexer → top-k → sparse MQA with attention sink + SWA.
- `hca_forward_kernel(H, params, cfg)` — HCA: heavy-compressor → dense MQA
  with sink + SWA.
- `sparse_attn_kernel(q, K_comp, topk_idxs, K_swa, sink)` — the inner MQA
  the reference TileLang `sparse_attn_kernel` ports over.
- `mhc_sinkhorn_kernel(...)` — the mHC Sinkhorn–Knopp projection.

All four are wired through `jax.custom_vjp`, so the agent can replace fwd
and bwd independently without touching call sites.

## Quick start

```bash
# Install (CPU dev). On a TPU host use `uv sync --extra tpu`.
uv sync

# Smoke test (CPU OK; tiny preset, ~1s)
uv run python bench.py --preset small_csa --seq 64

# Real shapes (TPU recommended; will be slow on CPU and OOM on big seqlens)
uv run python bench.py --preset dsv4_flash_csa --seq 16384 --bwd
uv run python bench.py --preset dsv4_pro_hca --seq 65536

# Sweep
uv run python bench.py --sweep
```

`bench.py` prints a single `---` block: `fwd_latency_ms`, `fwd_tflops`,
`fwd_mfu_percent`, `max_abs_diff` vs reference, `status`. The agent greps
this to record results.

## Running the agent

Spin up Claude/Codex in this repo (with permissions disabled), then:

```
Hi have a look at program.md and let's kick off a new kernel-dev experiment.
let's do the setup first.
```

`program.md` is the agent's working spec — what it can/can't touch, what
counts as a win, how to log results.

## Profiling

```bash
# JAX profiler trace (open in TensorBoard or chrome://tracing)
JAX_PLATFORMS=tpu uv run python bench.py --preset dsv4_flash_csa --seq 16384 \
    --profile /tmp/dsv4_trace

# HLO dumps
XLA_FLAGS="--xla_dump_to=/tmp/hlo --xla_dump_hlo_as_text" \
    uv run python bench.py --preset dsv4_flash_csa --seq 16384
```

Use `jax.named_scope("...")` in your kernel code to make the trace legible.

## Upstream

Branched from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
The original GPT-training driver (`prepare.py`, `train.py`, `analysis.ipynb`,
`progress.png`) is left in the tree for reference but is not used by this
kernel work.

## License

MIT
