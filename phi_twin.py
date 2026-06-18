"""Apples-to-apples: online-softmax vs phi, in the kernel's exact bf16 arithmetic.

Isolates the *algorithmic* precision difference (running-max vs fixed shift)
from the DMA/kernel machinery by reproducing the kernel's numeric path in plain
JAX: K dequantized to bf16, attention weights cast to bf16, dots accumulated in
fp32 — identical to attention.py. Both paths are graded against:
  - oracle  : fp32 dense softmax (serving_attn_ref math, exact)
  - bf16floor: the oracle rounded to bf16 (the irreducible error of a bf16 output)

Question answered: does phi degrade quality vs online, and by how much, relative
to the precision the bf16 output can even represent?
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dsv4.serving.attention import serving_attn_ref
from phi_verify import gather_logits_and_valid, make_inputs


def _weights_bf16_path(inp, mode, C):
    """Return normalized attention output using the kernel's bf16 arithmetic.

    mode='online': subtract per-row max (weights<=1, top weight exactly 1.0).
    mode='phi'   : subtract constant C (pure accumulation, no running max).
    """
    logits, valid, K_all = gather_logits_and_valid(inp)      # fp32 logits
    sink = inp["attn_sink"].astype(jnp.float32)[None, None, :, None]
    vmask = valid[:, :, None, :]
    K_bf = K_all.astype(jnp.bfloat16).astype(jnp.float32)    # dequant->bf16 (as kernel)

    neg = jnp.float32(-1e30)
    lg = jnp.where(vmask, logits, neg)
    if mode == "online":
        m = jnp.maximum(jnp.max(lg, -1, keepdims=True), sink)
        w = jnp.where(vmask, jnp.exp(lg - m), 0.0)
        denom = w.sum(-1, keepdims=True) + jnp.exp(sink - m)
    else:  # phi
        w = jnp.where(vmask, jnp.exp(lg - C), 0.0)
        denom = w.sum(-1, keepdims=True) + jnp.exp(sink - C)

    if mode == "phi_fp32w":
        w_op = w                                             # keep fp32 weights
    else:
        w_op = w.astype(jnp.bfloat16).astype(jnp.float32)    # weight cast (as kernel)
    num = jnp.einsum("bths,btsc->bthc", w_op, K_bf)
    return (num / denom).astype(jnp.bfloat16).astype(jnp.float32)  # bf16 output


def maxabs(a, b):
    return float(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)).max())


def main():
    print("online vs phi in the kernel's bf16 arithmetic, graded vs fp32 oracle")
    print("(bf16floor = oracle rounded to bf16 = the best a bf16 output can do)\n")
    print(f"{'seed':>4} | {'bf16floor':>10} | {'online':>10} | "
          f"{'phi bf16w':>10} | {'phi fp32w':>10} | {'phi C=0':>10}")
    print("-" * 70)

    agg = {k: 0.0 for k in ("floor", "online", "phi_sqrt", "phi_fp32w", "phi_0")}
    for s in range(8):
        inp = make_inputs(s)
        c = float(inp["c"])
        orc = serving_attn_ref(
            inp["q"], inp["kc"], inp["topk_idxs"], inp["swa"], inp["q_pos"],
            inp["attn_sink"], n_win=inp["n_win"], rope_dim=inp["rope_dim"]
        ).astype(jnp.float32)
        floor = maxabs(orc.astype(jnp.bfloat16), orc)
        online = maxabs(_weights_bf16_path(inp, "online", 0.0), orc)
        p_sqrt = maxabs(_weights_bf16_path(inp, "phi", c ** 0.5), orc)
        p_f32 = maxabs(_weights_bf16_path(inp, "phi_fp32w", c ** 0.5), orc)
        p_0 = maxabs(_weights_bf16_path(inp, "phi", 0.0), orc)
        for k, v in zip(agg, (floor, online, p_sqrt, p_f32, p_0)):
            agg[k] = max(agg[k], v)
        print(f"{s:>4} | {floor:>10.3e} | {online:>10.3e} | "
              f"{p_sqrt:>10.3e} | {p_f32:>10.3e} | {p_0:>10.3e}")

    print("-" * 70)
    print(f"{'WORST':>4} | {agg['floor']:>10.3e} | {agg['online']:>10.3e} | "
          f"{agg['phi_sqrt']:>10.3e} | {agg['phi_fp32w']:>10.3e} | {agg['phi_0']:>10.3e}")
    print(f"\nphi bf16w / online   = {agg['phi_sqrt']/agg['online']:.2f}x"
          f"   (cost of dropping the running max, bf16 weights)")
    print(f"phi fp32w / online   = {agg['phi_fp32w']/agg['online']:.2f}x"
          f"   (fp32 PV weights — recovers precision, QK dot still bf16)")
    print(f"phi bf16w / bf16floor = {agg['phi_sqrt']/agg['floor']:.2f}x")


if __name__ == "__main__":
    main()
