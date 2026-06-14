"""Verification harness for the phi-based (decoupled) softmax vs online softmax.

Run from repo root:  python phi_verify.py

Proves, before touching the kernel:
  1. BOUND   — max|logit| stays under sqrt(c) (Cauchy-Schwarz on RMSNormed q,k),
               so a *fixed* shift C replaces the online running max safely.
  2. MATH    — a decoupled fp32 phi-accumulation (numerator/denominator, no
               running max) reproduces the fp32 softmax oracle to fp32 noise.
  3. STABLE  — choosing C in {sqrt(c), 0, -bound} changes nothing in fp32
               (the normalizer cancels the shift exactly); only the accumulator
               magnitude moves, well within fp32 range.

After the kernel is implemented, `phi_verify_kernel.py` reuses these inputs to
compare the phi *kernel* against the online kernel and the oracle.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dsv4 import eager
from dsv4.serving import quant
from dsv4.serving.attention import serving_attn_ref, sparse_mqa_gathered
from dsv4.serving.config import tiles_for


# --------------------------------------------------------------------------
# Input builder — dev shapes, faithful to the kernel's contract
# --------------------------------------------------------------------------
def make_inputs(seed, *, B=2, T=3, n_h=4, c=128, rope_dim=64,
                S_c=32, topk=8, n_win=16, s_raw=64, gain=1.0):
    """RMSNormed q + RMSNormed/quantized KV, exactly as the runtime feeds them.

    `gain` scales q,k AFTER rms_norm to emulate a learned RMSNorm gain != 1
    (stress test for the bound). gain=1.0 is the real eager.rms_norm (no gain).
    """
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed), 5)

    q = jax.random.normal(k0, (B, T, n_h, c), jnp.float32)
    q = eager.rms_norm(q) * gain                       # [B,T,n_h,c], unit-RMS*gain

    kc_raw = eager.rms_norm(jax.random.normal(k1, (B, S_c, c), jnp.float32)) * gain
    sw_raw = eager.rms_norm(jax.random.normal(k2, (B, s_raw, c), jnp.float32)) * gain
    kc = quant.quantize_kv(kc_raw, rope_dim)
    swa = quant.quantize_kv(sw_raw, rope_dim)

    # distinct top-k per (b,t); poke a couple of -1 holes to exercise masking
    perm = jnp.argsort(jax.random.normal(k3, (B, T, S_c)), axis=-1)
    topk_idxs = perm[..., :topk].astype(jnp.int32)
    topk_idxs = topk_idxs.at[:, 0, -1].set(-1)         # one masked slot

    # positions late enough that the whole SWA window is in-range (non-ring)
    q_pos = jnp.full((B, T), s_raw - 1, jnp.int32) - jnp.arange(T, dtype=jnp.int32)[None, :]
    attn_sink = jax.random.normal(k4, (n_h,), jnp.float32) * 0.5

    return dict(q=q, kc=kc, swa=swa, topk_idxs=topk_idxs, q_pos=q_pos,
                attn_sink=attn_sink, n_win=n_win, rope_dim=rope_dim, c=c)


# --------------------------------------------------------------------------
# Reconstruct the exact (b,t,head,key) logit set the oracle sees, to measure
# the bound and to drive the phi math reference.
# --------------------------------------------------------------------------
def gather_logits_and_valid(inp):
    q, kc, swa = inp["q"], inp["kc"], inp["swa"]
    topk_idxs, q_pos = inp["topk_idxs"], inp["q_pos"]
    n_win, c = inp["n_win"], inp["c"]
    scale = float(c) ** -0.5

    K_comp = quant.dequantize_kv(kc)
    K_raw = quant.dequantize_kv(swa)
    safe = jnp.maximum(topk_idxs, 0)
    K_sel = jax.vmap(lambda kb, ib: kb[ib])(K_comp, safe)
    sel_valid = topk_idxs >= 0

    w_off = jnp.arange(n_win) - (n_win - 1)
    w_pos = q_pos[..., None] + w_off[None, None, :]
    w_safe = jnp.clip(w_pos, 0, K_raw.shape[1] - 1)
    K_win = jax.vmap(lambda kb, ib: kb[ib])(K_raw, w_safe)
    win_valid = w_pos >= 0

    K_all = jnp.concatenate([K_sel, K_win], axis=2)
    valid = jnp.concatenate([sel_valid, win_valid], axis=2)
    logits = jnp.einsum("bthc,btsc->bths", q.astype(jnp.float32), K_all) * scale
    return logits, valid, K_all


def phi_ref_fp32(inp, C):
    """Decoupled phi softmax in fp32: NO running max, single normalize.

    phi = exp(logit - C);  O = (Σ phi·V + 0·sink) / (Σ phi + exp(sink - C)).
    This is the math the kernel will implement. Must equal the oracle.
    """
    logits, valid, K_all = gather_logits_and_valid(inp)
    sink = inp["attn_sink"].astype(jnp.float32)[None, None, :, None]
    vmask = valid[:, :, None, :]

    phi = jnp.where(vmask, jnp.exp(logits - C), 0.0)         # [B,T,n_h,S]
    denom = phi.sum(-1, keepdims=True) + jnp.exp(sink - C)   # + sink term
    return jnp.einsum("bths,btsc->bthc", phi, K_all) / denom


def stats(name, a, b):
    a, b = a.astype(jnp.float32), b.astype(jnp.float32)
    abs_err = jnp.abs(a - b)
    denom = jnp.maximum(jnp.abs(b), 1e-6)
    rel = (abs_err / denom)
    # cosine similarity over the head/feature vector per token
    af, bf = a.reshape(-1), b.reshape(-1)
    cos = float(jnp.dot(af, bf) / (jnp.linalg.norm(af) * jnp.linalg.norm(bf)))
    print(f"  {name:<34} max|Δ|={float(abs_err.max()):.3e}  "
          f"mean|Δ|={float(abs_err.mean()):.3e}  "
          f"max_rel={float(rel.max()):.3e}  cos={cos:.8f}")


# --------------------------------------------------------------------------
def main():
    tiles = tiles_for(interpret=True)
    print(f"interpret={tiles.interpret}  attn_chunk={tiles.attn_chunk}\n")

    # ---- (1) BOUND: sweep seeds, measure max|logit| vs sqrt(c) -----------
    print("=== (1) logit bound: max|logit| vs theoretical sqrt(c) ===")
    for c in (128, 512):
        worst = 0.0
        for s in range(8):
            inp = make_inputs(s, c=c, rope_dim=64 if c == 128 else 64,
                              S_c=32, topk=8)
            lg, valid, _ = gather_logits_and_valid(inp)
            lg = jnp.where(valid[:, :, None, :], lg, 0.0)
            worst = max(worst, float(jnp.abs(lg).max()))
        bound = c ** 0.5
        print(f"  c={c:<4} sqrt(c)={bound:6.2f}   observed max|logit|={worst:6.3f}"
              f"   exp(bound)={jnp.exp(jnp.float32(bound)):.3e}  (fp32 max 3.4e38)")

    # ---- (2) MATH: phi fp32 vs oracle, across C choices ------------------
    print("\n=== (2) phi-math (fp32, no running max) vs softmax oracle ===")
    inp = make_inputs(0)
    oracle = serving_attn_ref(
        inp["q"], inp["kc"], inp["topk_idxs"], inp["swa"], inp["q_pos"],
        inp["attn_sink"], n_win=inp["n_win"], rope_dim=inp["rope_dim"])
    for C in (float(inp["c"]) ** 0.5, 0.0, -(float(inp["c"]) ** 0.5)):
        phi = phi_ref_fp32(inp, C)
        stats(f"phi_fp32(C={C:+.2f}) vs oracle", phi, oracle)

    # ---- (3) phi KERNEL vs oracle — must stay in the bf16 output envelope --
    # The kernel returns bf16. A bf16 output O(1) cannot represent better than
    # ~4e-3 per element (its own rounding floor, quantified in phi_twin.py).
    # PASS = worst max|Δ| within a standard bf16 attention atol. The apples-to-
    # apples online-vs-phi comparison (phi = 1.34x online abs err, both at the
    # bf16 floor) lives in phi_twin.py; here we only gate the absolute envelope.
    BF16_ATOL = 1.2e-2
    print("\n=== (3) phi KERNEL (current attention.py) vs oracle, seed sweep ===")
    worst = 0.0
    for s in range(6):
        inp_s = make_inputs(s)
        orc = serving_attn_ref(
            inp_s["q"], inp_s["kc"], inp_s["topk_idxs"], inp_s["swa"],
            inp_s["q_pos"], inp_s["attn_sink"], n_win=inp_s["n_win"],
            rope_dim=inp_s["rope_dim"])
        ker = sparse_mqa_gathered(
            inp_s["q"], inp_s["kc"], inp_s["topk_idxs"], inp_s["swa"],
            inp_s["q_pos"], inp_s["attn_sink"], n_win=inp_s["n_win"],
            rope_dim=inp_s["rope_dim"], tiles=tiles)
        d = float(jnp.abs(ker.astype(jnp.float32) - orc.astype(jnp.float32)).max())
        worst = max(worst, d)
        # parity: kernel-vs-oracle should equal kernel-vs-phi_fp32 (phi==softmax),
        # proving the kernel implements phi exactly (error is all quantization).
        dphi = float(jnp.abs(ker.astype(jnp.float32)
                     - phi_ref_fp32(inp_s, float(inp_s["c"]) ** 0.5)).max())
        print(f"  seed {s}: phi_kernel vs oracle max|Δ|={d:.3e}   "
              f"vs phi_fp32 max|Δ|={dphi:.3e}")
    verdict = "PASS" if worst <= BF16_ATOL else "FAIL"
    print(f"  -> worst max|Δ|={worst:.3e}  (bf16 atol {BF16_ATOL:.1e})  "
          f"[{verdict}: phi within the bf16 output envelope]")

    # ---- stress: emulate a learned RMSNorm gain > 1 ----------------------
    print("\n=== (stress) RMSNorm gain>1 inflates the bound — pick C accordingly ===")
    for g in (1.0, 2.0, 4.0):
        inp_g = make_inputs(0, gain=g)
        lg, valid, _ = gather_logits_and_valid(inp_g)
        lg = jnp.where(valid[:, :, None, :], lg, 0.0)
        mx = float(jnp.abs(lg).max())
        # phi math still exact if exp doesn't overflow: need mx - C < ~88
        C = float(inp_g["c"]) ** 0.5 * g
        oracle_g = serving_attn_ref(
            inp_g["q"], inp_g["kc"], inp_g["topk_idxs"], inp_g["swa"],
            inp_g["q_pos"], inp_g["attn_sink"], n_win=inp_g["n_win"],
            rope_dim=inp_g["rope_dim"])
        phi_g = phi_ref_fp32(inp_g, C)
        err = float(jnp.abs(phi_g - oracle_g).max())
        print(f"  gain={g}: max|logit|={mx:6.2f}  C={C:6.2f}  "
              f"phi_fp32 vs oracle max|Δ|={err:.3e}")


if __name__ == "__main__":
    main()
