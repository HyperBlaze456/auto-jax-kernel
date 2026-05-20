"""V2 Pallas-TPU kernel surface for DeepSeek V4 hybrid attention.

What V1 covered and what V2 adds
--------------------------------

``kernel_v1`` provides exactly one Pallas kernel: ``sparse_attn_kernel_v1``
— FlashAttention-with-sink on a pre-materialized ``K_full``. The full
``csa_forward_kernel`` / ``hca_forward_kernel`` / ``mhc_sinkhorn_kernel``
surface in ``dsv4/kernel.py`` still routed to the eager reference.

V2 closes that gap. The headline additions:

  1. ``csa_forward_kernel_v2`` — full CSA forward (compressor + lightning
     indexer + top-k + RoPE + RMSNorm in JAX, sparse-MQA core in Pallas).
     The JAX preamble stays in JAX because XLA already lowers those GEMMs
     and softmaxes well; the win is replacing the eager attention with
     FlashAttention's online softmax.
  2. ``hca_forward_kernel_v2`` — full HCA forward. Routes the dense-MQA
     core through the *same* Pallas kernel as CSA, by building a
     "select-all blocks with block-causal mask" ``topk_idxs`` (mirroring
     ``reference.hca_forward``'s ``n_blk <= 4096`` branch). This swaps the
     [B, n, n_h, n_blk] attention-weights materialization the reference
     does for FlashAttention's running ``(m, l, acc)``.
  3. ``mhc_sinkhorn_kernel_v2`` — per-token Pallas kernel for the
     Sinkhorn iteration loop. Keeps the hc×hc matrix in VMEM scratch
     across iterations rather than round-tripping HBM.
  4. ``sparse_attn_kernel_v2`` — surface-compatible re-export of
     ``kernel_v1.sparse_attn_kernel_v1`` so ``dsv4/kernel.py`` only
     imports one module.

Numerics (kernel_refs.md §I)
----------------------------

V2 introduces no new approximations. Online softmax keeps (m, l) in fp32;
QK^T and PV go through the MXU with ``preferred_element_type=jnp.float32``.
Expected ``max_abs_diff`` vs the fp32 reference at bf16 inputs is in the
1e-3 to 5e-3 range, the same as V1.

TPU-generation portability (kernel_refs.md §C/§J, kernel_config.py)
-------------------------------------------------------------------

Block sizes are resolved per (TPU generation, problem shape) by
``dsv4.kernel_config.config_for``. Adding a new generation only requires
registering a new ``TpuSpec``; this file does not encode any
generation-specific constants. The Sinkhorn tile size lives in
``KernelConfig.bn_sinkhorn``; the attention tiles use ``bq`` / ``bs`` (as
in v1).

Backward
--------

V2 carries hand-written Pallas backwards for the two ops where they
matter for memory/compute:

  - ``sparse_attn_kernel_v2`` inherits V1's saved-(lse) FlashAttention
    backward (kernel_refs §D3). Re-exporting V1's kernel means CSA and
    HCA pick the bwd up for free through natural autodiff over the JAX
    preamble (compressor, indexer, top-k, RoPE, RMSNorm).
  - ``mhc_sinkhorn_kernel_v2`` has a dedicated Pallas bwd that runs
    ``jax.vjp`` over a pure inner forward (matching the reference) for
    the 19-iter sinkhorn body. That keeps the hc×hc residual stack in
    VMEM rather than re-materializing 19 intermediate matrices in HBM.
    The affine outer step (``comb_init = comb_raw · hc_scale[2] +
    comb_bias``) and the pre/post sigmoid paths fall out via closed-form
    JAX outside the kernel — they don't need Pallas residuals.

CSA and HCA themselves no longer need a custom_vjp at this layer: their
JAX preamble is regular GEMM/softmax that XLA autodiffs natively, and
the only Pallas op in the chain (``sparse_attn_kernel_v2``) brings its
own custom_vjp from V1.

What's still V3+ (kernel_refs §D3, paper §3.3)
----------------------------------------------

  - Per-program deterministic KV-grad scratch for streaming bwd across
    very long S. V1's bwd writes dk per (b, qi, si) tile, which is
    already deterministic because no two programs target the same slot;
    the §3.3 pattern only matters once a streaming bwd shares K entries
    across programs.
  - Streaming HCA dense path (n_blk > 4096). Still raises; same
    limitation as the reference.
"""

from __future__ import annotations

from functools import lru_cache, partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from . import reference as ref
from .kernel_config import KernelConfig, default_config
from .kernel_v1 import sparse_attn_kernel_v1


# ---------------------------------------------------------------------------
# sparse_attn_kernel_v2 — re-export of v1's Pallas core
# ---------------------------------------------------------------------------
#
# V1 already lifts the CSA sparse MQA core into a generation-agnostic Pallas
# kernel. V2 reuses it as-is; the surface name change is purely so
# ``dsv4/kernel.py`` has one consistent import root.

def sparse_attn_kernel_v2(
    q: jax.Array,
    K_comp: jax.Array,
    topk_idxs: jax.Array,
    K_swa: jax.Array,
    attn_sink: jax.Array,
    *,
    config: KernelConfig | None = None,
) -> jax.Array:
    """V2 forward for the CSA sparse-MQA core (delegates to v1's kernel)."""
    return sparse_attn_kernel_v1(
        q, K_comp, topk_idxs, K_swa, attn_sink, config=config
    )


# ---------------------------------------------------------------------------
# CSA full forward (Pallas attention core + JAX preamble)
# ---------------------------------------------------------------------------
#
# The structure mirrors ``reference.csa_forward`` exactly; only the final
# ``ref.sparse_attn_with_sink`` call is swapped for the Pallas kernel.

def _csa_forward_v2(
    H: jax.Array, p: ref.CSAParams, cfg: ref.CSAConfig, *, config: KernelConfig,
) -> jax.Array:
    B, n, d = H.shape

    with jax.named_scope("csa_compress"):
        K_comp = ref.csa_compress(
            H, p.W_aKV, p.W_bKV, p.W_aZ, p.W_bZ, p.B_a, p.B_b, cfg.m,
        )                                                       # [B, n/m, c]
        K_IComp = ref.csa_compress(
            H, p.W_aIK, p.W_bIK, p.W_aIZ, p.W_bIZ, p.B_aI, p.B_bI, cfg.m,
        )                                                       # [B, n/m, c_I]

    with jax.named_scope("csa_indexer"):
        scores = ref.lightning_indexer(
            H, K_IComp, p.W_DQ, p.W_IUQ, p.W_w, cfg.m, cfg.n_I_h, cfg.c_I,
        )
        topk_idxs = ref.topk_indices(scores, cfg.topk)          # [B, n, k]

    with jax.named_scope("csa_qkv_proj"):
        cQ = H @ p.W_DQ
        q = (cQ @ p.W_UQ).reshape(B, n, cfg.n_h, cfg.c)         # [B, n, n_h, c]
        K_swa_full = H @ p.W_swaK                               # [B, n, c]

    with jax.named_scope("csa_rope"):
        cos_q, sin_q = ref._build_rope(n, cfg.rope_dim)
        q = ref.apply_partial_rope(
            q, cos_q[None, :, None, :], sin_q[None, :, None, :], cfg.rope_dim,
        )
        n_blk = K_comp.shape[1]
        cos_k, sin_k = ref._build_rope(n_blk, cfg.rope_dim)
        K_comp = ref.apply_partial_rope(
            K_comp, cos_k[None, :, :], sin_k[None, :, :], cfg.rope_dim,
        )
        cos_s, sin_s = ref._build_rope(n, cfg.rope_dim)
        K_swa_full = ref.apply_partial_rope(
            K_swa_full, cos_s[None, :, :], sin_s[None, :, :], cfg.rope_dim,
        )

    with jax.named_scope("csa_swa_gather"):
        K_swa = ref.swa_gather(K_swa_full, cfg.n_win)           # [B, n, n_win, c]

    with jax.named_scope("csa_rmsnorm"):
        q = ref.rms_norm(q)
        K_comp = ref.rms_norm(K_comp)
        K_swa = ref.rms_norm(K_swa)

    with jax.named_scope("csa_sparse_mqa"):
        return sparse_attn_kernel_v2(
            q, K_comp, topk_idxs, K_swa, p.attn_sink, config=config,
        )


# ---------------------------------------------------------------------------
# HCA full forward (Pallas attention core + JAX preamble)
# ---------------------------------------------------------------------------
#
# Reference.hca_forward has two branches: ``n_blk <= 4096`` builds a
# select-all ``topk_idxs`` and routes through ``sparse_attn_with_sink``;
# anything larger raises NotImplementedError because a streaming dense
# kernel hasn't been written yet. V2 keeps that contract — the select-all
# path now flows through the *Pallas* sparse_attn (huge win, since the
# reference's [B, n, n_h, n_blk] softmax weights tensor is the main
# memory hog at scale) but the streaming path is still V3+ territory.

def _hca_forward_v2(
    H: jax.Array, p: ref.HCAParams, cfg: ref.HCAConfig, *, config: KernelConfig,
) -> jax.Array:
    B, n, d = H.shape

    with jax.named_scope("hca_compress"):
        K_comp = ref.hca_compress(H, p.W_KV, p.W_Z, p.B, cfg.m_prime)
    n_blk = K_comp.shape[1]

    with jax.named_scope("hca_qkv_proj"):
        cQ = H @ p.W_DQ
        q = (cQ @ p.W_UQ).reshape(B, n, cfg.n_h, cfg.c)
        K_swa_full = H @ p.W_swaK

    with jax.named_scope("hca_rope"):
        cos_q, sin_q = ref._build_rope(n, cfg.rope_dim)
        q = ref.apply_partial_rope(
            q, cos_q[None, :, None, :], sin_q[None, :, None, :], cfg.rope_dim,
        )
        cos_k, sin_k = ref._build_rope(n_blk, cfg.rope_dim)
        K_comp = ref.apply_partial_rope(
            K_comp, cos_k[None, :, :], sin_k[None, :, :], cfg.rope_dim,
        )
        cos_s, sin_s = ref._build_rope(n, cfg.rope_dim)
        K_swa_full = ref.apply_partial_rope(
            K_swa_full, cos_s[None, :, :], sin_s[None, :, :], cfg.rope_dim,
        )

    with jax.named_scope("hca_swa_gather"):
        K_swa = ref.swa_gather(K_swa_full, cfg.n_win)

    with jax.named_scope("hca_rmsnorm"):
        q = ref.rms_norm(q)
        K_comp = ref.rms_norm(K_comp)
        K_swa = ref.rms_norm(K_swa)

    if n_blk > 4096:
        raise NotImplementedError(
            f"HCA dense path with n_blk={n_blk} > 4096 needs a streaming "
            f"Pallas implementation (V3+). The reference has the same "
            f"limitation; reduce sequence length or m_prime to test."
        )

    # Build select-all topk_idxs with block-level causality, then route
    # through the same Pallas sparse_attn kernel CSA uses.
    with jax.named_scope("hca_build_causal_idxs"):
        all_idx = jnp.broadcast_to(
            jnp.arange(n_blk, dtype=jnp.int32), (B, n, n_blk),
        )
        t_idx = jnp.arange(n)
        causal = jnp.arange(n_blk)[None, :] <= (t_idx[:, None] // cfg.m_prime)
        causal = jnp.broadcast_to(causal[None, :, :], (B, n, n_blk))
        all_idx = jnp.where(causal, all_idx, jnp.int32(-1))

    with jax.named_scope("hca_dense_mqa"):
        return sparse_attn_kernel_v2(
            q, K_comp, all_idx, K_swa, p.attn_sink, config=config,
        )


# ---------------------------------------------------------------------------
# mHC Sinkhorn (per-token Pallas kernel)
# ---------------------------------------------------------------------------
#
# Reference does ``softmax(comb) → row/col normalise → 19 iterations``,
# all in JAX. V2 lifts that into a Pallas kernel so the hc×hc matrix stays
# in VMEM scratch across iterations instead of round-tripping HBM each
# step. The grid is (B, n // BN); within each program we run the full
# iteration loop on BN tokens.
#
# Layout note (kernel_refs §J #1, §J #3): the mixes input is split into
# three reshape-free pytree leaves *outside* the kernel — pre_raw, post_raw,
# comb_raw (already reshaped to [..., hc, hc]). That keeps every block
# slice along contiguous axes, so we never trip the "reshape touching the
# last two dims" restriction inside the kernel.

def _sinkhorn_body(
    pre_raw_ref,      # [1, BN, hc]
    post_raw_ref,     # [1, BN, hc]
    comb_raw_ref,     # [1, BN, hc, hc]
    hc_scale_ref,     # [3]
    pre_bias_ref,     # [hc]
    post_bias_ref,    # [hc]
    comb_bias_ref,    # [hc, hc]
    pre_ref,          # [1, BN, hc]   out
    post_ref,         # [1, BN, hc]   out
    comb_ref,         # [1, BN, hc, hc] out
    *,
    sinkhorn_iters: int,
    eps: float,
):
    pre_raw = pre_raw_ref[...].astype(jnp.float32)
    post_raw = post_raw_ref[...].astype(jnp.float32)
    comb_raw = comb_raw_ref[...].astype(jnp.float32)
    hc_scale = hc_scale_ref[...].astype(jnp.float32)
    pre_bias = pre_bias_ref[...].astype(jnp.float32)
    post_bias = post_bias_ref[...].astype(jnp.float32)
    comb_bias = comb_bias_ref[...].astype(jnp.float32)

    pre = jax.nn.sigmoid(pre_raw * hc_scale[0] + pre_bias) + jnp.float32(eps)
    post = 2.0 * jax.nn.sigmoid(post_raw * hc_scale[1] + post_bias)
    comb = comb_raw * hc_scale[2] + comb_bias  # broadcasts over (1, BN)

    # First pass: row-softmax + col-normalise (matches reference exactly).
    comb = jax.nn.softmax(comb, axis=-1) + jnp.float32(eps)
    comb = comb / (comb.sum(axis=-2, keepdims=True) + jnp.float32(eps))

    def step(_, c):
        c = c / (c.sum(axis=-1, keepdims=True) + jnp.float32(eps))
        c = c / (c.sum(axis=-2, keepdims=True) + jnp.float32(eps))
        return c

    comb = jax.lax.fori_loop(0, sinkhorn_iters - 1, step, comb)

    pre_ref[...] = pre.astype(pre_ref.dtype)
    post_ref[...] = post.astype(post_ref.dtype)
    comb_ref[...] = comb.astype(comb_ref.dtype)


def _sinkhorn_pallas(
    pre_raw: jax.Array,    # [B, n, hc]
    post_raw: jax.Array,   # [B, n, hc]
    comb_raw: jax.Array,   # [B, n, hc, hc]
    hc_scale: jax.Array,   # [3]
    pre_bias: jax.Array,   # [hc]
    post_bias: jax.Array,  # [hc]
    comb_bias: jax.Array,  # [hc, hc]
    *,
    sinkhorn_iters: int,
    eps: float,
    config: KernelConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    B, n, hc = pre_raw.shape
    bn = config.bn_sinkhorn
    if n % bn != 0:
        raise ValueError(f"n={n} must be a multiple of BN_sinkhorn={bn}; pad in the wrapper")

    grid = (B, n // bn)
    pre_raw_spec  = pl.BlockSpec((1, bn, hc),     lambda b, ni: (b, ni, 0))
    post_raw_spec = pl.BlockSpec((1, bn, hc),     lambda b, ni: (b, ni, 0))
    comb_raw_spec = pl.BlockSpec((1, bn, hc, hc), lambda b, ni: (b, ni, 0, 0))
    scale_spec    = pl.BlockSpec((3,),            lambda b, ni: (0,))
    pre_bias_spec = pl.BlockSpec((hc,),           lambda b, ni: (0,))
    post_bias_spec = pl.BlockSpec((hc,),          lambda b, ni: (0,))
    comb_bias_spec = pl.BlockSpec((hc, hc),       lambda b, ni: (0, 0))
    pre_out_spec  = pl.BlockSpec((1, bn, hc),     lambda b, ni: (b, ni, 0))
    post_out_spec = pl.BlockSpec((1, bn, hc),     lambda b, ni: (b, ni, 0))
    comb_out_spec = pl.BlockSpec((1, bn, hc, hc), lambda b, ni: (b, ni, 0, 0))

    out_shapes = (
        jax.ShapeDtypeStruct((B, n, hc), pre_raw.dtype),
        jax.ShapeDtypeStruct((B, n, hc), post_raw.dtype),
        jax.ShapeDtypeStruct((B, n, hc, hc), comb_raw.dtype),
    )

    return pl.pallas_call(
        partial(_sinkhorn_body, sinkhorn_iters=sinkhorn_iters, eps=eps),
        grid=grid,
        in_specs=[
            pre_raw_spec, post_raw_spec, comb_raw_spec,
            scale_spec, pre_bias_spec, post_bias_spec, comb_bias_spec,
        ],
        out_specs=[pre_out_spec, post_out_spec, comb_out_spec],
        out_shape=out_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
        interpret=config.interpret,
    )(pre_raw, post_raw, comb_raw,
      hc_scale, pre_bias, post_bias, comb_bias)


# ---------------------------------------------------------------------------
# mHC Sinkhorn backward (per-token Pallas kernel for the 19-iter loop only)
# ---------------------------------------------------------------------------
#
# The bwd splits cleanly into three pieces:
#
#   (1) Pre / post sigmoid path: closed-form JAX outside the kernel.
#       ``pre = sigmoid(z_pre) + eps`` with ``z_pre = pre_raw · hc_scale[0] +
#       pre_bias`` — sigmoid bwd is a one-line ``sig · (1 - sig)`` multiply.
#       No iterative residuals to save, so no Pallas win.
#
#   (2) Comb sinkhorn iterations: Pallas. The 19-step row/col-normalize
#       loop needs intermediate residuals to autodiff. Running ``jax.vjp``
#       over a pure forward (mirroring ``reference.mhc_sinkhorn``'s scan)
#       inside a Pallas kernel keeps those residuals in VMEM scratch.
#
#   (3) Affine outer step on comb (``comb_init = comb_raw · hc_scale[2] +
#       comb_bias``): closed-form JAX outside. The kernel outputs the
#       cotangent on ``comb_init``; the wrapper turns that into
#       ``dcomb_raw``, ``dhc_scale[2]``, ``dcomb_bias``.
#
# Why pure-function + jax.vjp inside Pallas rather than a hand-derived
# analytic bwd for the 19 iterations: the row/col-normalize step has a
# non-trivial Jacobian (``y = x / (sum(x) + eps)``), and writing it out
# explicitly times 19 iterations is a lot of code with no clear win over
# what jax.vjp already produces — both end up storing the per-iter ``c``
# in VMEM. The bwd kernel is computationally what kernel_refs §D3
# advocates (saved-residual recompute), just expressed via jax.vjp.


def _sinkhorn_norm_pure(comb_init, *, sinkhorn_iters, eps):
    """Pure forward for the post-affine comb normalization path.

    Mirrors ``reference.mhc_sinkhorn``'s scan-based iteration (NOT the
    Pallas forward's ``fori_loop`` — fori_loop is non-differentiable,
    while scan natively supports reverse-mode autodiff). The two paths
    are numerically identical for the same iteration count.
    """
    comb = jax.nn.softmax(comb_init, axis=-1) + jnp.float32(eps)
    comb = comb / (comb.sum(axis=-2, keepdims=True) + jnp.float32(eps))

    def step(c, _):
        c = c / (c.sum(axis=-1, keepdims=True) + jnp.float32(eps))
        c = c / (c.sum(axis=-2, keepdims=True) + jnp.float32(eps))
        return c, None

    comb, _ = jax.lax.scan(step, comb, None, length=sinkhorn_iters - 1)
    return comb


def _sinkhorn_norm_body_bwd(
    comb_init_ref,    # [1, BN, hc, hc]  bf16/f32  — in
    dcomb_ref,        # [1, BN, hc, hc]  bf16/f32  — in (upstream)
    dcomb_init_ref,   # [1, BN, hc, hc]  bf16/f32  — out (cotangent on comb_init)
    *,
    sinkhorn_iters: int,
    eps: float,
):
    comb_init = comb_init_ref[...].astype(jnp.float32)
    dcomb = dcomb_ref[...].astype(jnp.float32)

    _, vjp_fn = jax.vjp(
        partial(_sinkhorn_norm_pure, sinkhorn_iters=sinkhorn_iters, eps=eps),
        comb_init,
    )
    (dcomb_init,) = vjp_fn(dcomb)
    dcomb_init_ref[...] = dcomb_init.astype(dcomb_init_ref.dtype)


def _sinkhorn_norm_pallas_bwd(
    comb_init: jax.Array,   # [B, n, hc, hc]
    dcomb: jax.Array,       # [B, n, hc, hc]
    *,
    sinkhorn_iters: int,
    eps: float,
    config: KernelConfig,
) -> jax.Array:
    """Returns ``dcomb_init`` of the same shape as ``comb_init``."""
    B, n, hc, _ = comb_init.shape
    bn = config.bn_sinkhorn
    if n % bn != 0:
        raise ValueError(f"n={n} must be a multiple of BN_sinkhorn={bn}; pad in the wrapper")

    grid = (B, n // bn)
    spec_4d = pl.BlockSpec((1, bn, hc, hc), lambda b, ni: (b, ni, 0, 0))

    return pl.pallas_call(
        partial(_sinkhorn_norm_body_bwd, sinkhorn_iters=sinkhorn_iters, eps=eps),
        grid=grid,
        in_specs=[spec_4d, spec_4d],
        out_specs=spec_4d,
        out_shape=jax.ShapeDtypeStruct(comb_init.shape, comb_init.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
        interpret=config.interpret,
    )(comb_init, dcomb)


def _mhc_sinkhorn_v2_bwd(
    res, douts, *, hc: int, sinkhorn_iters: int, eps: float, config: KernelConfig,
):
    """Custom-vjp backward for ``_mhc_sinkhorn_v2``.

    Returns ``(dmixes, dhc_scale, dhc_base)`` matching the forward's
    diff'd inputs. The pre/post and affine outer comb paths are
    closed-form in JAX; only the 19-iter sinkhorn loop goes through
    Pallas.
    """
    mixes, hc_scale, hc_base = res
    dpre, dpost, dcomb = douts
    B, n, _ = mixes.shape

    # Split / reshape — mirrors the forward.
    pre_raw = mixes[..., :hc]                               # [B, n, hc]
    post_raw = mixes[..., hc:2 * hc]                         # [B, n, hc]
    comb_raw = mixes[..., 2 * hc:].reshape(B, n, hc, hc)     # [B, n, hc, hc]
    pre_bias = hc_base[:hc]
    post_bias = hc_base[hc:2 * hc]
    comb_bias = hc_base[2 * hc:].reshape(hc, hc)

    # --- (1) Pre/post sigmoid bwd, all in JAX. ---
    pre_raw_f = pre_raw.astype(jnp.float32)
    post_raw_f = post_raw.astype(jnp.float32)
    hc_scale_f = hc_scale.astype(jnp.float32)

    z_pre = pre_raw_f * hc_scale_f[0] + pre_bias.astype(jnp.float32)
    sig_pre = jax.nn.sigmoid(z_pre)
    dz_pre = dpre.astype(jnp.float32) * sig_pre * (1.0 - sig_pre)   # [B, n, hc]
    dpre_raw = (dz_pre * hc_scale_f[0]).astype(mixes.dtype)
    dpre_bias = dz_pre.sum(axis=(0, 1))                              # [hc]
    dhc_scale_0 = (dz_pre * pre_raw_f).sum()

    z_post = post_raw_f * hc_scale_f[1] + post_bias.astype(jnp.float32)
    sig_post = jax.nn.sigmoid(z_post)
    dz_post = dpost.astype(jnp.float32) * 2.0 * sig_post * (1.0 - sig_post)
    dpost_raw = (dz_post * hc_scale_f[1]).astype(mixes.dtype)
    dpost_bias = dz_post.sum(axis=(0, 1))                            # [hc]
    dhc_scale_1 = (dz_post * post_raw_f).sum()

    # --- (2) Comb sinkhorn bwd, Pallas. ---
    comb_init = (comb_raw.astype(jnp.float32) * hc_scale_f[2]
                 + comb_bias.astype(jnp.float32))                    # [B, n, hc, hc]
    # Match the forward's dtype to keep the kernel's compute path consistent.
    comb_init = comb_init.astype(mixes.dtype)
    dcomb_padded = dcomb.astype(mixes.dtype)

    bn = config.bn_sinkhorn
    pad_n = (-n) % bn
    if pad_n:
        comb_init = jnp.pad(comb_init, ((0, 0), (0, pad_n), (0, 0), (0, 0)))
        dcomb_padded = jnp.pad(dcomb_padded, ((0, 0), (0, pad_n), (0, 0), (0, 0)))

    dcomb_init_padded = _sinkhorn_norm_pallas_bwd(
        comb_init, dcomb_padded,
        sinkhorn_iters=sinkhorn_iters, eps=eps, config=config,
    )

    if pad_n:
        dcomb_init = dcomb_init_padded[:, :n]
    else:
        dcomb_init = dcomb_init_padded
    dcomb_init_f = dcomb_init.astype(jnp.float32)

    # --- (3) Affine outer step bwd on comb, JAX. ---
    comb_raw_f = comb_raw.astype(jnp.float32)
    dcomb_raw = (dcomb_init_f * hc_scale_f[2]).astype(mixes.dtype)   # [B, n, hc, hc]
    dcomb_bias = dcomb_init_f.sum(axis=(0, 1))                        # [hc, hc]
    dhc_scale_2 = (dcomb_init_f * comb_raw_f).sum()

    # Reassemble dmixes (concat the three split parts back into the last dim)
    # and dhc_base (concat the three bias parts into the flat hc_base layout).
    dmixes = jnp.concatenate(
        [dpre_raw, dpost_raw, dcomb_raw.reshape(B, n, hc * hc)],
        axis=-1,
    )
    dhc_base = jnp.concatenate(
        [dpre_bias, dpost_bias, dcomb_bias.reshape(hc * hc)],
        axis=0,
    ).astype(hc_base.dtype)
    dhc_scale = jnp.stack([dhc_scale_0, dhc_scale_1, dhc_scale_2]).astype(hc_scale.dtype)

    return dmixes, dhc_scale, dhc_base


def _mhc_sinkhorn_v2(
    mixes: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    hc: int,
    sinkhorn_iters: int,
    eps: float,
    *,
    config: KernelConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    B, n, mixes_dim = mixes.shape
    if mixes_dim != (2 + hc) * hc:
        raise ValueError(
            f"mixes last dim = {mixes_dim} but expected (2 + hc) * hc = "
            f"{(2 + hc) * hc} for hc={hc}"
        )

    # Split + reshape outside the kernel — this is a no-op slicing on the
    # caller's side, and it avoids in-kernel reshapes that would touch the
    # last two block dims (kernel_refs §J #3).
    pre_raw = mixes[..., :hc]                               # [B, n, hc]
    post_raw = mixes[..., hc:2 * hc]                         # [B, n, hc]
    comb_raw = mixes[..., 2 * hc:].reshape(B, n, hc, hc)     # [B, n, hc, hc]
    pre_bias = hc_base[:hc]
    post_bias = hc_base[hc:2 * hc]
    comb_bias = hc_base[2 * hc:].reshape(hc, hc)

    # Pad n to a multiple of BN_sinkhorn. The padded rows compute against
    # zero biases and dummy mixes (the math is well-defined either way),
    # and we slice them off below.
    bn = config.bn_sinkhorn
    pad_n = (-n) % bn
    if pad_n:
        pre_raw = jnp.pad(pre_raw, ((0, 0), (0, pad_n), (0, 0)))
        post_raw = jnp.pad(post_raw, ((0, 0), (0, pad_n), (0, 0)))
        comb_raw = jnp.pad(comb_raw, ((0, 0), (0, pad_n), (0, 0), (0, 0)))

    pre, post, comb = _sinkhorn_pallas(
        pre_raw, post_raw, comb_raw,
        hc_scale, pre_bias, post_bias, comb_bias,
        sinkhorn_iters=sinkhorn_iters, eps=eps, config=config,
    )

    if pad_n:
        pre = pre[:, :n]
        post = post[:, :n]
        comb = comb[:, :n]
    return pre, post, comb


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------
#
# CSA / HCA have no custom_vjp at this layer: the only Pallas op in their
# chain (``sparse_attn_kernel_v2``, re-exporting V1) brings its own
# hand-written custom_vjp from V1, and the rest of the preamble is plain
# JAX that XLA autodiffs natively. The lru_cache + closure pattern is no
# longer needed for these — JAX's jit cache handles trace reuse.
#
# Sinkhorn keeps a custom_vjp because ``pl.pallas_call`` is not
# autodifferentiable on TPU by default; the bwd is the hand-written
# Pallas kernel above (see ``_mhc_sinkhorn_v2_bwd``).


@lru_cache(maxsize=None)
def _make_sinkhorn_kernel(config: KernelConfig, hc: int, sinkhorn_iters: int, eps: float):
    @jax.custom_vjp
    def fn(mixes, hc_scale, hc_base):
        return _mhc_sinkhorn_v2(
            mixes, hc_scale, hc_base, hc, sinkhorn_iters, eps, config=config,
        )

    def _fwd(mixes, hc_scale, hc_base):
        out = _mhc_sinkhorn_v2(
            mixes, hc_scale, hc_base, hc, sinkhorn_iters, eps, config=config,
        )
        return out, (mixes, hc_scale, hc_base)

    def _bwd(res, douts):
        return _mhc_sinkhorn_v2_bwd(
            res, douts,
            hc=hc, sinkhorn_iters=sinkhorn_iters, eps=eps, config=config,
        )

    fn.defvjp(_fwd, _bwd)
    return fn


def csa_forward_kernel_v2(
    H: jax.Array,
    params: ref.CSAParams,
    cfg: ref.CSAConfig,
    *,
    config: KernelConfig | None = None,
) -> jax.Array:
    """End-to-end CSA forward via V2.

    JAX-handled preamble (compress, indexer, top-k, RoPE, RMSNorm); Pallas
    FlashAttention-with-sink for the sparse MQA core. Generation-agnostic
    via ``KernelConfig`` (auto-detected when ``config is None``).

    Backward is natural autodiff: the JAX preamble autodiffs natively;
    the sparse-MQA Pallas core's bwd comes from V1's hand-written
    ``custom_vjp`` (saved-lse FlashAttention bwd).
    """
    if config is None:
        config = default_config(n_h=cfg.n_h, c=cfg.c)
    return _csa_forward_v2(H, params, cfg, config=config)


def hca_forward_kernel_v2(
    H: jax.Array,
    params: ref.HCAParams,
    cfg: ref.HCAConfig,
    *,
    config: KernelConfig | None = None,
) -> jax.Array:
    """End-to-end HCA forward via V2.

    Reuses the same Pallas sparse-MQA kernel as CSA, with a select-all
    block-causal mask in place of top-k indices. Errors at n_blk > 4096
    (same as the reference) until a streaming Pallas variant lands in V3+.

    Backward path is the same as CSA: natural autodiff over the JAX
    preamble + V1's hand-written sparse-MQA bwd.
    """
    if config is None:
        config = default_config(n_h=cfg.n_h, c=cfg.c)
    return _hca_forward_v2(H, params, cfg, config=config)


def mhc_sinkhorn_kernel_v2(
    mixes: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    hc: int,
    sinkhorn_iters: int,
    eps: float,
    *,
    config: KernelConfig | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pallas mHC Sinkhorn projection (matches ``reference.mhc_sinkhorn``).

    Forward keeps the hc×hc matrix in VMEM across the 19-iter loop;
    backward (see ``_mhc_sinkhorn_v2_bwd``) does the same for the
    cotangent path via ``jax.vjp`` inside a second Pallas kernel.
    """
    if config is None:
        # hc/c values don't drive the sinkhorn tile budget; pick something
        # sane just to populate the rest of KernelConfig (bq/bs are unused
        # here but the dataclass needs them).
        config = default_config(n_h=1, c=128)
    return _make_sinkhorn_kernel(config, hc, sinkhorn_iters, eps)(
        mixes, hc_scale, hc_base,
    )
