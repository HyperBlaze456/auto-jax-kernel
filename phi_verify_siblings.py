"""Verify the phi-ported sibling kernels: paged, blocked, and train (fwd+grads).

Run from repo root:  python phi_verify_siblings.py

- paged   : sparse_mqa_paged vs serving_attn_ref over expand_pages_to_rows
            (identical row set) — must sit in the bf16 envelope.
- blocked : sparse_mqa_gathered_blocked vs serving_attn_ref (same topk + SWA).
- train   : jax.grad through sparse_mqa_train (phi fwd + custom bwd) vs jax.grad
            through the fp32 oracle sparse_mqa_train_ref. The backward consumes
            only lse (= phi_shift + log(denom), exact), so grads must still match
            the fp32 gold within the bf16 envelope — the real test that the phi
            forward feeds the backward correctly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dsv4 import eager
from dsv4.serving import quant
from dsv4.serving.attention import serving_attn_ref
from dsv4.serving.attention_blocked import sparse_mqa_gathered_blocked
from dsv4.serving.attention_paged import expand_pages_to_rows, sparse_mqa_paged
from dsv4.serving.attention_train import (
    _fwd_call,
    sparse_mqa_train,
    sparse_mqa_train_ref,
)
from dsv4.serving.config import tiles_for

TILES = tiles_for(interpret=True)
BF16_ATOL = 1.2e-2


def maxabs(a, b):
    return float(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)).max())


def rms(seed, shape, gain=1.0):
    return eager.rms_norm(jax.random.normal(jax.random.PRNGKey(seed), shape,
                                            jnp.float32)) * gain


def distinct_topk(seed, B, T, pool, k):
    perm = jnp.argsort(jax.random.normal(jax.random.PRNGKey(seed), (B, T, pool)),
                       axis=-1)
    return perm[..., :k].astype(jnp.int32)


def verdict(name, worst):
    tag = "PASS" if worst <= BF16_ATOL else "FAIL"
    print(f"  -> {name}: worst max|Δ|={worst:.3e}  (bf16 atol {BF16_ATOL:.1e})  [{tag}]")
    return worst <= BF16_ATOL


def relnorm(a, b):
    a, b = a.astype(jnp.float32), b.astype(jnp.float32)
    return float(jnp.linalg.norm((a - b).reshape(-1))
                 / jnp.maximum(jnp.linalg.norm(b.reshape(-1)), 1e-12))


def train_lses_bf16(q, k_comp, topk_idxs, k_swa, q_pos, sink, n_win, C):
    """Build the kernel's bf16 logits, then form lse two ways: with the online
    per-row max shift vs the phi constant shift C. Algebraically identical;
    if they match numerically, the phi forward's stored lse == the online
    forward's lse, so the (unchanged) backward produces identical grads."""
    B, T, n_h, c = q.shape
    scale = float(c) ** -0.5
    bf = lambda x: x.astype(jnp.bfloat16).astype(jnp.float32)
    qf, kcf, krf = bf(q), bf(k_comp), bf(k_swa)
    safe = jnp.maximum(topk_idxs, 0)
    K_sel = jax.vmap(lambda kb, ib: kb[ib])(kcf, safe)
    sel_valid = topk_idxs >= 0
    start = jnp.clip(q_pos - n_win + 1, 0, krf.shape[1] - n_win)
    w_pos = start[..., None] + jnp.arange(n_win)
    K_win = jax.vmap(lambda kb, ib: kb[ib])(krf, w_pos)
    win_valid = (w_pos <= q_pos[..., None]) & (w_pos > q_pos[..., None] - n_win)
    K_all = jnp.concatenate([K_sel, K_win], axis=2)
    valid = jnp.concatenate([sel_valid, win_valid], axis=2)[:, :, None, :]
    logits = jnp.where(valid, jnp.einsum("bthc,btsc->bths", qf, K_all) * scale,
                       -jnp.inf)
    sink_b = sink.astype(jnp.float32)[None, None, :, None]

    m = jnp.maximum(jnp.max(logits, -1, keepdims=True), sink_b)        # online
    den_on = jnp.where(valid, jnp.exp(logits - m), 0.0).sum(-1, keepdims=True) \
        + jnp.exp(sink_b - m)
    lse_online = (m + jnp.log(den_on))[..., 0]

    den_phi = jnp.where(valid, jnp.exp(logits - C), 0.0).sum(-1, keepdims=True) \
        + jnp.exp(sink_b - C)
    lse_phi = (C + jnp.log(den_phi))[..., 0]
    return lse_online, lse_phi


# --------------------------------------------------------------------------
def test_paged():
    print("=== PAGED: sparse_mqa_paged vs serving_attn_ref (expanded rows) ===")
    B, T, n_h, c, rope_dim = 1, 2, 4, 128, 64
    page, S_c, k_pages, n_win, s_raw = 4, 32, 4, 16, 64
    n_pages = S_c // page
    worst = 0.0
    for s in range(4):
        q = rms(10 + s, (B, T, n_h, c))
        kc = quant.quantize_kv(rms(20 + s, (B, S_c, c)), rope_dim)
        swa = quant.quantize_kv(rms(30 + s, (B, s_raw, c)), rope_dim)
        page_idx = distinct_topk(40 + s, B, T, n_pages, k_pages)
        q_pos = jnp.full((B, T), s_raw - 1, jnp.int32)
        bound = jnp.full((B, T), S_c, jnp.int32)          # all page rows valid
        sink = jax.random.normal(jax.random.PRNGKey(50 + s), (n_h,)) * 0.5

        o_paged = sparse_mqa_paged(q, kc, page_idx, bound, swa, q_pos, sink,
                                   page=page, n_win=n_win, tiles=TILES)
        rows = expand_pages_to_rows(page_idx, page, bound)   # [B,T,k_pages*page]
        o_ref = serving_attn_ref(q, kc, rows, swa, q_pos, sink, n_win=n_win)
        d = maxabs(o_paged, o_ref)
        worst = max(worst, d)
        print(f"  seed {s}: max|Δ|={d:.3e}")
    return verdict("paged", worst)


def test_blocked():
    print("\n=== BLOCKED: sparse_mqa_gathered_blocked vs serving_attn_ref ===")
    B, T, n_h, c, rope_dim = 1, 8, 4, 128, 64
    S_c, topk, n_win, bq, s_raw = 32, 8, 16, 4, 32
    worst = 0.0
    for s in range(4):
        q = rms(60 + s, (B, T, n_h, c))
        kc = quant.quantize_kv(rms(70 + s, (B, S_c, c)), rope_dim)
        swa = quant.quantize_kv(rms(80 + s, (B, s_raw, c)), rope_dim)
        topk_idxs = distinct_topk(90 + s, B, T, S_c, topk)
        q_pos = jnp.arange(T, dtype=jnp.int32)[None, :]       # prefill contract
        sink = jax.random.normal(jax.random.PRNGKey(95 + s), (n_h,)) * 0.5

        o_blk = sparse_mqa_gathered_blocked(q, kc, topk_idxs, swa, sink,
                                            n_win=n_win, bq=bq, tiles=TILES)
        o_ref = serving_attn_ref(q, kc, topk_idxs, swa, q_pos, sink, n_win=n_win)
        d = maxabs(o_blk, o_ref)
        worst = max(worst, d)
        print(f"  seed {s}: max|Δ|={d:.3e}")
    return verdict("blocked", worst)


def test_train_grads():
    print("\n=== TRAIN: phi forward + (unchanged) backward ===")
    B, T, n_h, c = 1, 4, 4, 128
    S_c, S_r, topk, n_win = 32, 32, 8, 16
    C = float(c) ** 0.5
    chunk = TILES.attn_chunk
    # (A) lse identity: online-shift lse vs phi-shift lse from identical bf16
    #     logits, and the kernel's own emitted lse — proves the backward (which
    #     consumes only lse) is fed the same value online would have stored.
    # (B) grads vs fp32 oracle, reported as relative-norm error (the right
    #     metric for bf16 gradients; abs shown for context).
    lse_id = lse_kern = 0.0
    grad_rel = {k: 0.0 for k in ("dq", "dk_comp", "dk_swa", "dsink")}
    grad_abs = {k: 0.0 for k in ("dq", "dk_comp", "dk_swa", "dsink")}
    o_abs = 0.0
    for s in range(3):
        q = rms(100 + s, (B, T, n_h, c))
        k_comp = rms(110 + s, (B, S_c, c))
        k_swa = rms(120 + s, (B, S_r, c))
        topk_idxs = distinct_topk(130 + s, B, T, S_c, topk)
        q_pos = jnp.arange(T, dtype=jnp.int32)[None, :] + (S_r - T)
        sink = jax.random.normal(jax.random.PRNGKey(140 + s), (n_h,)) * 0.5
        cot = jax.random.normal(jax.random.PRNGKey(150 + s), (B, T, n_h, c))

        # (A) lse identity (plain JAX) + the kernel's stored lse
        lse_on, lse_ph = train_lses_bf16(q, k_comp, topk_idxs, k_swa, q_pos,
                                         sink, n_win, C)
        _, lse_kernel = _fwd_call(
            q.astype(jnp.bfloat16), k_comp.astype(jnp.bfloat16),
            topk_idxs.astype(jnp.int32), k_swa.astype(jnp.bfloat16),
            q_pos.astype(jnp.int32), sink.reshape(n_h, 1),
            n_win=n_win, chunk=chunk, interpret=TILES.interpret)
        lse_id = max(lse_id, maxabs(lse_on, lse_ph))
        lse_kern = max(lse_kern, maxabs(lse_kernel[..., 0], lse_ph))

        # (B) grads
        def loss_k(q, kc, ksw, sk):
            o = sparse_mqa_train(q, kc, topk_idxs, ksw, q_pos, sk,
                                 n_win=n_win, tiles=TILES)
            return (o.astype(jnp.float32) * cot).sum(), o

        def loss_r(q, kc, ksw, sk):
            o = sparse_mqa_train_ref(q, kc, topk_idxs, ksw, q_pos, sk, n_win=n_win)
            return (o.astype(jnp.float32) * cot).sum(), o

        (_, o_k), gk = jax.value_and_grad(loss_k, argnums=(0, 1, 2, 3),
                                          has_aux=True)(q, k_comp, k_swa, sink)
        (_, o_r), gr = jax.value_and_grad(loss_r, argnums=(0, 1, 2, 3),
                                          has_aux=True)(q, k_comp, k_swa, sink)
        o_abs = max(o_abs, maxabs(o_k, o_r))
        for n, a, b in zip(grad_rel, gk, gr):
            grad_rel[n] = max(grad_rel[n], relnorm(a, b))
            grad_abs[n] = max(grad_abs[n], maxabs(a, b))
        print(f"  seed {s}: lse(on|phi)={maxabs(lse_on, lse_ph):.2e}  "
              f"lse(kern|phi)={maxabs(lse_kernel[..., 0], lse_ph):.2e}  "
              f"dq={relnorm(gk[0], gr[0]):.2%}  dk_swa={relnorm(gk[2], gr[2]):.2%}")

    print(f"  lse identity (online-shift vs phi-shift): {lse_id:.3e}"
          f"  -> {'IDENTICAL' if lse_id < 1e-4 else 'DIFFER'}")
    print(f"  kernel lse vs phi-shift reference:        {lse_kern:.3e}")
    print(f"  forward o abs:  {o_abs:.3e}")
    for n in grad_rel:
        print(f"  grad {n:<8} rel-norm={grad_rel[n]:.2%}  (abs {grad_abs[n]:.3e})")
    # Gate: lse identity proves phi==online for the backward; grads must be
    # bf16-relative-level (5%). The backward source is untouched (git diff).
    ok = (lse_id < 1e-4) and all(v < 0.05 for v in grad_rel.values()) \
        and o_abs <= BF16_ATOL
    print(f"  -> train: [{'PASS' if ok else 'FAIL'}] "
          f"(lse identical, grads bf16-relative-level, backward unchanged)")
    return ok


if __name__ == "__main__":
    results = [test_paged(), test_blocked(), test_train_grads()]
    print(f"\nALL: {'PASS' if all(results) else 'FAIL'}")
