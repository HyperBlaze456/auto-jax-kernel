# Kernel reference card

Curated code + facts for writing Pallas-TPU kernels and sharding DSv4-scale
models across pods. Read this *and* the spec (`DeepSeek_V4.pdf` §2.3) before
touching `dsv4/kernel.py`.

Every code block is taken from the JAX docs or `jax-ml/jax`. Source URL
follows each section. When in doubt, click through.

---

## §A. TPU hardware constants you must know

| Gen   | bf16 TF/s | HBM GB/s | VMEM | SMEM   | TensorCores |
|-------|-----------|----------|------|--------|-------------|
| v4    | 138       | 615      | 16M  | 1 MiB  | 2           |
| v5e   | 197       | 820      | 128M | 1 MiB  | 1           |
| v5p   | 230†      | 1230     | 64M  | 1 MiB  | 2           |
| v6e   | 920       | 1640     | 128M | 1 MiB  | 1           |

† scaling-book lists v5p at 459 TF/s per *chip* (= 2 cores × 230). Use 459
for chip-level MFU on v5p; use 230 for single-core kernels.

- **MXU**: 128×128 systolic array on v5p and earlier (v6e: 256×256). Native
  op `bf16[8,128] @ bf16[128,128] → f32[8,128]` every 8 cycles.
- **VPU (vector unit)**: shape **8 × 128** (sublanes × lanes). VREGs are
  8×128 tiles. Last two dims of every array get tiled into VREGs.
- **ICI bandwidth**: v5p 9.0e10 B/s per axis (one-way), 1.8e11 B/s bidir.
  v5e is half that.
- **Arithmetic intensity threshold (ICI)**: ~2550 on v5p — TP becomes
  comms-bound past that.

URLs: `jax-ml.github.io/scaling-book/tpus`, `docs.jax.dev/.../pallas/tpu/hardware.html`

---

## §B. Pallas API surface — keep open while coding

| API | Role |
|-----|------|
| `pl.pallas_call(kernel, out_shape, grid, in_specs, out_specs)` | Lift kernel into JAX op |
| `pl.BlockSpec(block_shape, index_map_fn)` | Per-program block slice + memory space |
| `pl.program_id(axis)` | Current grid coord |
| `pl.num_programs(axis)` | Grid size on axis |
| `pl.ds(start, size)` / `pl.dslice(...)` | Dynamic slice |
| `ref[idx]` → load, `ref[idx] = v` → store | Ref indexing (bare `pl.load`/`pl.store` were removed; mask ragged/oob with `jnp.where`) |
| `pl.when(cond)` | Conditional block (decorates a `def _():`) |
| `pltpu.PrefetchScalarGridSpec(num_scalar_prefetch, grid, in_specs, out_specs, scratch_shapes)` | Grid that passes scalar arrays to index_map |
| `pltpu.VMEM(shape, dtype)` / `pltpu.SMEM(...)` | Explicit scratch in a memory space |
| `pltpu.emit_pipeline(kernel, grid, in_specs, out_specs)` | Inner-loop pipelining inside a kernel |
| `pltpu.make_async_copy(src, dst, sem)` | HBM↔VMEM async copy |
| `pltpu.make_async_remote_copy(src, dst, send_sem, recv_sem, device_id, device_id_type)` | Cross-chip DMA |
| `pltpu.SemaphoreType.{DMA,REGULAR,BARRIER}` | Sem kinds |
| `pltpu.get_barrier_semaphore()` | Global barrier |
| `pltpu.CompilerParams(dimension_semantics=("parallel",), collective_id=N)` | Per-call hints |

**Ref vs. Array**: kernel arguments are `Ref`s (mutable buffers). `ref[...]`
loads to registers/VMEM; `ref[...] = v` stores. Slice-indexing a Ref returns
a `jax.Array`; scalar-indexing returns an address. `Refs` cannot flow into
arbitrary JAX primitives — read first.

URLs: `docs.jax.dev/.../pallas/quickstart.html`, `pallas/design/design.html`

---

## §C. Pallas-TPU memory + layout — the SRAM-rolling-out playbook

### Layout rules (don't fight these)

- **Last two dims** of every block tile into VREGs as 8×128. Singleton
  dimensions on the last two axes waste compute. Keep last-dim ≥ 128 if
  possible.
- **Block shape constraint**: last two block dims must be **multiples of 8
  and 128**, or equal to the full array dims.
- **Reshapes** that touch the last two dims are restricted. Plan layout up
  front; transpose at HBM boundaries, not inside the K-loop.
- **Reductions over the last dim are slowest**; reduce over leading axes
  when you can choose. Same for broadcasts.

### Memory spaces

```python
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

# In a BlockSpec, you pick the memory space the block lives in:
pl.BlockSpec(shape, index_map, memory_space=pltpu.MemorySpace.VMEM)   # default
pl.BlockSpec(shape, index_map, memory_space=pl.ANY)                   # leaves in HBM (pltpu.ANY removed; use pl.ANY or pltpu.MemorySpace.HBM)
# scratch (persistent across grid iterations):
scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32),
                pltpu.SMEM((1024,), jnp.int32)]
```

- **VMEM**: main on-chip working memory. 64–128 MiB. Hold blocks of Q/K/V,
  accumulators here. Overflow → compiler error.
- **SMEM**: scalar memory, 1 MiB. **Put control-flow data here** (loop
  bounds, indices, top-k arrays read by `index_map`).
- **HBM**: where args land if `memory_space=pltpu.ANY`. High latency. The
  pipeline machinery moves blocks to VMEM for you.

### Scratch as accumulator pattern

```python
def matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps):
  @pl.when(pl.program_id(2) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  acc_ref[...] += jnp.dot(
      x_ref[...], y_ref[...], preferred_element_type=jnp.float32
  )

  @pl.when(pl.program_id(2) == nsteps - 1)
  def _():
    z_ref[...] = acc_ref[...].astype(z_ref.dtype)

# call: grid=(m//bm, n//bn, k//bk), scratch_shapes=[pltpu.VMEM((bm,bn), jnp.float32)]
```

Default tile sizes that win on v5p: **bm=512, bk=1024, bn=1024 → 80–90 % MFU
on dense matmul.** Start there, shrink if VMEM overflows.

URLs: `docs.jax.dev/.../pallas/tpu/details.html`, `pallas/tpu/matmul.html`

---

## §D. Canonical kernel patterns

### D1. Vector add (the "hello world")

```python
def add_kernel(x_ref, y_ref, o_ref):
  o_ref[...] = x_ref[...] + y_ref[...]

@jax.jit
def add(x, y):
  return pl.pallas_call(add_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x, y)
```

### D2. FlashAttention forward — the online-softmax recipe

This is **the** reference for our `sparse_attn_kernel`. Keys: running max
`m`, running denom `l`, accumulator `acc`. On each K-block:

```python
# from jax/experimental/pallas/ops/tpu/flash_attention.py
m_prev = m_scratch_ref[batch_idx]
l_prev = l_scratch_ref[batch_idx]
q = q_tile_ref[batch_idx]                                  # [block_q, head_dim]
k = k_tile_ref[..., pl.dslice(start_k, block_k), :]
s = jax.lax.dot_general(q, k, TRANS_B_DIM_NUMBERS)          # QK^T
# (apply mask + bias + scale here)

m_curr = jnp.max(s, axis=1)[:, None]
m_next = jnp.maximum(m_prev, m_curr)
p      = jnp.exp(s - jnp.tile(m_next, (1, block_k_repeats)))
alpha  = jnp.exp(m_prev - m_next)                           # rescale prev
l_corr = alpha * l_prev
l_next = jnp.sum(p, axis=1)[:, None] + l_corr

# rescale running output, add new contribution
l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
acc_scratch_ref[batch_idx] *= l_broadcast(l_corr * l_next_inv_safe)
o_curr = jax.lax.dot(p.astype(v.dtype), v, ...)
acc_scratch_ref[batch_idx] += o_curr * l_broadcast(l_next_inv_safe)
```

Block sizes: `block_q=128, block_k=128, block_b=1`. Inner K-loop unrolled
via `@pl.loop(0, block_k_major, step=block_k, unroll=True)`.

For DSv4 specifically:
- Replace per-K-block dense gather with **scalar-prefetched top-k** (see D4).
- Add **attention sink**: at the end of the K-loop, account for
  `exp(sink_h - m_final)` in the denominator before the final divide.
- Apply **partial RoPE on last 64 dims** before computing `s`.

### D3. Backward via saved residuals

`_flash_attention_bwd_dkv` and `_flash_attention_bwd_dq` reload `q,k,v` and
the saved `m,l` to recompute `p`:

```python
capped_logits = lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, ...)
p  = jnp.exp(capped_logits - jnp.tile(m, (1, block_k // MIN_BLOCK_SIZE)))
p  = p * jnp.tile(1 / l, (1, block_k // MIN_BLOCK_SIZE))
dv = lax.dot(p.T.astype(do.dtype), do, ...)
# dQ accumulated via dQ_scratch; dK,dV accumulated similarly
```

DSv4 nuance (paper §3.3): the V4 inference kernel uses **per-SM accumulation
buffers + a global deterministic sum** instead of `atomicAdd` for KV grads.
On TPU, allocate one scratch buffer per program in the sequence axis and
sum across them in a final pass — same idea, different hardware.

### D4. **Block-sparse with scalar prefetch** ← directly applicable to top-k MQA

Pass an int32 array of selected block indices to the index_map via
`PrefetchScalarGridSpec.num_scalar_prefetch=N`. Each grid iteration's
`index_map` receives the prefetched scalars and returns the block coord
into the dense KV that this query needs.

```python
def dsd_kernel(idxs_i_ref, idxs_k_ref,                      # prefetched
               x_ref, y_ref, _, o_ref,                      # in / out
               accum_scratch):                              # scratch
  blk_idx = pl.program_id(1)
  is_start = blk_idx == 0
  changed = (idxs_i_ref[blk_idx] != idxs_i_ref[jnp.maximum(blk_idx-1, 0)])

  @pl.when(is_start | changed)
  def _():
    accum_scratch[...] = jnp.zeros_like(accum_scratch)

  accum_scratch[...] += jnp.dot(x_ref[0, :, :], y_ref[...],
                                preferred_element_type=jnp.float32)

  next_change = (idxs_i_ref[blk_idx] !=
                 idxs_i_ref[jnp.minimum(blk_idx+1, num_blocks)])
  is_end = blk_idx == (num_blocks - 1)

  @pl.when(is_end | next_change)
  def _():
    o_ref[...] = accum_scratch[...].astype(o_ref.dtype)


def x_map(j, blk_idx, blk_idxs_i, blk_idxs_k):
  return (blk_idx, 0, 0)
def y_map(j, blk_idx, blk_idxs_i, blk_idxs_k):
  return (blk_idxs_k[blk_idx], j)            # ← read from prefetched scalar
def o_map(j, blk_idx, blk_idxs_i, blk_idxs_k):
  return (blk_idxs_i[blk_idx], j)


grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=2,
    grid=(N // blk_N, num_blocks),
    in_specs=[pl.BlockSpec((1, blk_M, blk_K), x_map),
              pl.BlockSpec((blk_K, blk_N),  y_map),
              pl.BlockSpec((blk_M, blk_N),  o_map)],
    out_specs=pl.BlockSpec((blk_M, blk_N), o_map),
    scratch_shapes=[pltpu.VMEM((blk_M, blk_N), jnp.float32)],
)
```

Reported speedups: **6× at 10 % sparsity** (SpMM), **1.8× at 50 % mask**
(masked dense). For our top-k=512 / S=n/m blocks ratio (DSv4-Flash at 64K
seq → S=16384, k/S=3 %), this pattern is the right starting point.

URL: `docs.jax.dev/.../pallas/tpu/sparse.html`

---

## §E. Pipelining and multi-buffering

```python
# Per-input buffering count
pl.BlockSpec(pipeline_mode=pl.Buffered(buffer_count=2))            # default
pl.BlockSpec(pipeline_mode=pl.Buffered(buffer_count=2,
                                       use_lookahead=True))         # variable compute
```

For **nested pipelines** (e.g. ICI all-gather + HBM→VMEM stream):

```python
def inner_kernel(input_ref, accum_ref):
  ...    # small VMEM-friendly work

accum_pipeline = pltpu.emit_pipeline(
    inner_kernel,
    in_specs=[inner_block_spec],
    out_specs=inner_block_spec,
    grid=inner_grid,
)
accum_pipeline(input_arg, output_arg)
```

**Megacore (v4/v5p has 2 cores per chip)**:

```python
pl.pallas_call(kernel, grid=(2,),
    compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)))
```

Mark embarrassingly parallel grid axes `"parallel"` to split across both
cores. Default is sequential lex order.

URL: `docs.jax.dev/.../pallas/tpu/pipelining.html`

---

## §F. Distributed Pallas — collectives written *inside* a kernel

You write the collective yourself when the compiler's auto version isn't
overlapping the way you want, or when the collective fuses into a kernel
(reduce-scatter inside the matmul epilogue, etc.).

### Cross-chip DMA primitive

```python
remote = pltpu.make_async_remote_copy(
    src_ref=input_ref,
    dst_ref=output_ref,
    send_sem=send_sem,
    recv_sem=recv_sem,
    device_id=(right_neighbor,),
    device_id_type=pl.DeviceIdType.MESH,    # tuple into mesh
)
remote.start()
remote.wait()
```

### Right-permute (one-hop ring step)

```python
def right_permute_kernel(input_ref, output_ref, send_sem, recv_sem):
  my_id = jax.lax.axis_index('x')
  right = jax.lax.rem(my_id + 1, num_devices)
  op = pltpu.make_async_remote_copy(
      input_ref, output_ref, send_sem, recv_sem,
      device_id=(right,), device_id_type=pl.DeviceIdType.MESH)
  op.start(); op.wait()
```

### All-gather ring with double-buffering

```python
# scratch: 2 DMA sems + (num_devices-1)-element recv sem array
scratch_shapes=([pltpu.SemaphoreType.DMA] * 2
                + [pltpu.SemaphoreType.DMA((num_devices-1,))])
# grid drives ring iterations: grid=(num_devices-1,)
# inside kernel:
iteration       = pl.program_id(0)
working_slot    = jax.lax.rem(iteration, 2)
receiving_slot  = 1 - working_slot
# write to neighbor's receiving_slot, read from own working_slot
```

### Local async copy (HBM↔VMEM)

```python
op = pltpu.make_async_copy(src_ref=input_ref, dst_ref=output_ref, sem=local_sem)
op.start(); op.wait()
```

### Barrier

```python
barrier = pltpu.get_barrier_semaphore()
# requires compiler_params=pltpu.CompilerParams(collective_id=0)
```

### shard_map + pallas_call composition

```python
result = jax.jit(jax.shard_map(
    kernel_fn,
    mesh=mesh,
    in_specs=part,
    out_specs=part,
    check_vma=False,
))(input_arr)
```

Inside the kernel, `jax.lax.axis_index('x')` gives the device coordinate.

URL: `docs.jax.dev/.../pallas/tpu/distributed.html`

---

## §G. shard_map for outer-level sharding

### Signature

```python
jax.shard_map(f, /, *, out_specs, in_specs=jax.sharding.Infer,
              mesh=None, axis_names=frozenset({}), check_vma=True) → F
```

Body sees per-device shapes: `sz // mesh.shape[axis_name]` for axes named
in the spec, full size for unmentioned axes.

### Mesh + spec construction

```python
from jax.sharding import Mesh, PartitionSpec as P
mesh = jax.make_mesh((4, 2), ('x', 'y'))     # 8 devices, 2D mesh
jax.set_mesh(mesh)
```

### Manual matmul w/ `psum`

```python
@jax.shard_map(in_specs=(P('x','y'), P('y', None)), out_specs=P('x', None))
def matmul_basic(a_block, b_block):
  c_partial = jnp.dot(a_block, b_block)
  return jax.lax.psum(c_partial, 'y')
```

### Reduce-scatter epilogue

```python
@jax.shard_map(in_specs=(P('i','j'), P('j', None)), out_specs=P('i','j'))
def matmul_rs(a_block, b_block):
  c_partial = jnp.matmul(a_block, b_block)
  return jax.lax.psum_scatter(c_partial, 'j', scatter_dimension=1, tiled=True)
```

### All-gather, all-to-all, psum_scatter

```python
y = jax.lax.all_gather(x, 'i')          # varying → varying
y = jax.lax.psum_scatter(x, 'i')        # varying → varying
y = jax.lax.all_to_all(x, 'i', 0, 0)    # varying → varying  (cheap, 1/4 cost of all-gather)
```

### When to use shard_map vs. jit-with-sharding

> "It is almost always simpler to write a program in `jit==pjit` — but if a
> given part of the program is less optimized by the compiler than it could
> be, drop into `shmap`!" — JEP 14273

Drop in for: TP overlap, expert-parallel routing, custom collective
schedules, anywhere you want the per-device view explicit.

URLs: `docs.jax.dev/.../notebooks/shard_map.html`, `jep/14273-shard-map.html`,
`_autosummary/jax.shard_map.html`

---

## §H. Sharding patterns at DSv4 scale

### Communication primitives + costs

| Op            | Bytes-moved cost                         |
|---------------|------------------------------------------|
| AllGather     | `bytes / (ICI_bw · num_axes)`            |
| ReduceScatter | same as AllGather                        |
| AllReduce     | **2× AllGather** (= AG + RS)             |
| AllToAll      | **AllGather / 4**                        |

ICI arithmetic intensity threshold (v5p): **2550** — past this you become
comms-bound on TP. DCN ≈ 71360.

### Four matmul-sharding cases (memorize)

```
A[I_X, J ] · B[J , K_Y] → C[I_X, K_Y]                    # case 1: free, no comms
A[I  , J_X] · B[J , K  ] → AllGather_X(A) then matmul    # case 2a: ag-input
                          → matmul → AllReduce_X         # case 2b: ar-output (cheaper if K << J)
A[I  , J_X] · B[J_X, K ] → matmul → AllReduce_X / RS     # case 3: contract-shared
A[I_X, J ] · B[J , K_X] → AllGather one input first      # case 4
```

### Strategy table for transformer

```
DP        : In[B_X, D] · W_in[D, F] · W_out[F, D] → Out[B_X, D]
FSDP/Z3   : In[B_X, D] · W_in[D_X, F] · W_out[F, D_X] → Out[B_X, D]
TP        : In[B, D_Y] · W_in[D, F_Y] · W_out[F_Y, D] → Out[B, D_Y]
FSDP+TP   : In[B_X, D_Y] · W_in[D_X, F_Y] · W_out[F_Y, D_X] → Out[B_X, D_Y]
```

FSDP forward: `AllGather(W_in[D_X, F])` → matmul → `AllGather(W_out[F, D_X])`
→ matmul. Backward: replace AllReduce on grads with ReduceScatter. The
`AllGather + ReduceScatter == AllReduce` identity is the trick.

### MoE / expert parallelism (the AllToAll trick)

Tokens scattered across devices need to land on the device hosting their
chosen expert. AllToAll reshapes from `[device, expert_X]` → `[device_X, expert]`.
**Cost: 1/4 of AllGather on a ring** — that's why MoE uses it.

```python
# inside shard_map body, after routing decisions:
tokens_per_expert = jax.lax.all_to_all(tokens, 'expert_axis', 0, 0)
# now each device has the tokens it owns experts for
```

For DSv4 (256–384 routed experts, top-6 per token), expect AllToAll on
both dispatch and combine. The V4 paper's "MegaMoE" kernel (§3.1) fuses
both AllToAll's with the GEMMs into one mega-kernel; conceptually that's
two `pltpu.make_async_remote_copy` waves bracketing the linear-1/linear-2
matmuls inside one pallas_call.

### Attention head parallelism

```
Q[B, H_X, D] · K[B, H_X, D]^T → A[B, H_X, S, S]   # no comms — heads are independent
```

Shard along `n_h` (or `n_kv_head` for GQA). For DSv4-Pro's 128 query heads
this is the obvious axis. Combine with FSDP on D for the projection
weights.

### Compute-bound thresholds (v5p, scaling-book)

- DP/FSDP: comms-bound when `B/X < 2550`
- TP: comms-bound when `Y > F/2550 · M_Y`
- FSDP + TP: optimal `X_opt = √((B/F) · (M_X/M_Y) · N)`

URLs: `jax-ml.github.io/scaling-book/sharding`, `.../training`

---

## §I. Numerics

- **Default**: bf16 inputs, **fp32 accumulator**. Scratch must be f32:
  `pltpu.VMEM((bm, bn), jnp.float32)`. Use `preferred_element_type=jnp.float32`
  on `jnp.dot` / `lax.dot_general` to force the bf16×bf16→f32 MXU path.
- **Online softmax** must keep `m, l` in fp32 even if Q/K/V are bf16.
- **bf16 inputs against fp32 reference** → expect `max_abs_diff` around
  **1e-3 to 5e-3** at typical sequence lengths. That's not a regression.
- For **bitwise-determinism** (paper §3.3), abandon split-K, use one
  accumulator per program then deterministic global sum across them.

---

## §J. Footguns (the ones that bite first)

1. **Last two block dims** must be multiples of 8 and 128 (or full).
   Violations → confusing compiler errors.
2. **Reductions on the last dim are 5–10× slower** than on leading dims.
   Plan layout so reductions are on lane-major axes.
3. **Reshapes touching the last two dims** are restricted. Plan layout
   once at the HBM boundary.
4. **VMEM overflow** is silent at trace time, loud at compile. Sum your
   block buffer sizes × buffer_count and stay under generation cap.
5. **`jax.grad` through Pallas can transpose your access pattern** and
   force atomics. Use `jax.custom_vjp` for any kernel you care about.
6. **Loop primitives unroll** during compilation. Keep static trip counts
   small or use the `unroll=False` knob if the kernel grows huge.
7. **Cross-device-id types**: `pl.DeviceIdType.MESH` (tuple) vs. `LOGICAL`
   (int). Get this wrong and DMAs go to the wrong place silently.
8. **Reuse of recv semaphores** across simultaneous DMAs deadlocks. Use
   `pltpu.SemaphoreType.DMA((num_devices-1,))` arrays.
9. **`jnp.sin`, `jnp.cos`** are expensive on TPU — precompute RoPE cos/sin
   tables in HBM at JIT time, not inside the kernel.
10. **Small batch + large grid** wastes pipeline depth. Aim for grids ≥ 4
    on the parallel axis so multi-buffering pays off.

---

## §K. Profiling — how to actually find the bottleneck

```bash
# JAX profiler (TensorBoard / Perfetto)
JAX_PLATFORMS=tpu python bench.py --preset dsv4_flash_csa --seq 16384 \
    --profile /tmp/dsv4_trace

# HLO dump (post-optimization IR)
XLA_FLAGS="--xla_dump_to=/tmp/hlo --xla_dump_hlo_as_text" python bench.py ...

# TPU MFU table (peak bf16 TF/s) lives in bench.py — pass --peak-tflops to override
```

Use `jax.named_scope("csa_compress")`, `jax.named_scope("indexer")`,
`jax.named_scope("sparse_mqa")` etc. so the trace is legible.

`pltpu.CompilerParams(collective_id=N)` tags a kernel for trace/HLO
correlation — useful when you have several Pallas kernels in one program.

XProf-on-TPU surfaces: MXU utilization, VMEM utilization, HBM bandwidth,
ICI bandwidth, kernel duration. If MXU < 30 % the kernel is bandwidth- or
launch-bound; if HBM > 80 % you're memory-bound (good — means MXU idle is
the ceiling, fix by raising arithmetic intensity).

---

## §L. Cheat-sheet for our specific kernels

### `sparse_attn_kernel` (CSA core MQA)

- Pattern: **§D2 (FlashAttention forward) + §D4 (scalar-prefetched indices)**.
- Prefetch `topk_idxs[B, n, k]` as int32. Index map for K_comp uses
  `prefetch_idxs[query_idx, j]` to pick which compressed block to gather.
- Tile along `(B, n, n_h)` parallel; sequential along `(k + n_win)` inner.
- Block shapes to try: `block_q=128`, `block_k=64` (matches the V4 TileLang
  reference's `block=64` setting), `block_n_h=8`.
- Custom backward (§D3) saves `m, l` per query; recompute `p`; accumulate
  KV grads with deterministic per-program scratch then sum.

### `csa_compress`

- Per-block softmax-mix over **2m** positions, c-channels. 2m=8 is small;
  this is reduction-heavy on `c` (the last dim). Layout: keep `c` as the
  lane axis so the softmax broadcast is leading-dim. Compress in fp32,
  cast back at the end.

### `lightning_indexer` + top-k

- The matmul `qI · K_IComp.T` is just a GEMM (§D1/§C scratch-accumulator
  pattern). The ReLU + head-mix is a fused vector op after the GEMM.
- **Top-k**: use `jax.lax.top_k` — there's no Pallas-TPU primitive for it;
  bring out to JAX, run on the indexer scores, then prefetch the result
  into the next kernel.

### `hca_forward_kernel`

- Same as §D2 but with **all** compressed entries (no top-k) and a causal
  block-mask. No scalar prefetch needed; classic FlashAttention.

### `mhc_sinkhorn_kernel`

- Per-token (B, n) parallel grid; `hc × hc` matrix lives entirely in
  registers. `T.serial(sinkhorn_iters)` translates to a `jax.lax.fori_loop`
  with `unroll=True` if `sinkhorn_iters` is small (e.g. 20). Reductions
  over `hc` (small dim) — fine on TPU. Don't bother sharding this; tokens
  are independent.

### Distributed shape (DSv4-Pro at 1M seq)

- **Mesh suggestion**: 4D `(dp, fsdp, tp, ep)` over a v5p pod.
- Attention: head-parallel along `tp` (128 heads), FSDP weights along
  `fsdp`. Sequence sharded along the same axis as `dp` for context
  parallelism (CP) at very long contexts (paper §3.5.3).
- MoE: expert-parallel along `ep`, AllToAll on dispatch/combine.
- Use `shard_map` with `check_vma=False` around each `pallas_call` so
  collectives inside the kernel are explicit.

---

*Updated: 2026-06-01 (API surface re-checked against jax/jaxlib 0.10.1).
Source URLs are fingerprintable; if a doc has moved upstream, search
`docs.jax.dev` for the section title.*
