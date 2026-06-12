"""Roofline estimate: DSv4 theoretical max serving throughput on TPU v5e–v7.

Produces the numbers in HARDWARE_NOTES.md §11. Two regimes:

``analyze``: the compute ceiling — decode at the large-batch limit, where
weight streaming amortizes to ~0/token and only the per-token terms
(gather, indexer scan, ICI) remain.

``decode_memory_bound``: the regime serving actually lives in — per-step
HBM bytes as a function of per-chip batch ``b``:

    bytes/step = dense weights (bf16, replicated read every step)
               + shared expert (fp8, every step)
               + E[distinct routed experts hit] x per-expert fp8 bytes
               + b x (KV gather + indexer scan + HCA reads)
               + b x mHC residual-stream traffic

Expected expert hits: each of the b·n_chips global tokens picks topk of E
uniformly (roofline assumption; real routing is more skewed = fewer
distinct experts = fewer bytes), so a chip holding E/n experts streams
(E/n)·(1 - (1 - topk/E)^(b·n)) of them per layer per step.

Common machinery:

- Exact parameter counts via ``jax.eval_shape`` over ``init_params``
  (nothing is materialized; Pro is ~1.6T params). Activated params =
  attention/mHC/router/head + shared expert + topk/n_routed of routed.
- Static FLOPs/token = 2 x activated matmul params, split fp8 (MoE expert
  GEMMs) vs bf16 (everything else) so chips without native fp8 MXU price
  them at the bf16 rate.
- Context terms per decode token: gather flash QK+PV over (topk + n_win)
  rows, the indexer scan over all n_blk compressed keys (FLOPs *and* its
  bf16 ki-cache read), HCA's dense pass over S/m' entries.

Run: ``python -m dsv4.serving.throughput_roofline``
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .config import DSV4_FLASH, DSV4_PRO, ModelConfig
from .model import init_params, layer_schedule

ROW_BYTES = 580  # 448 fp8 nope + 64*2 rope bf16 + 4 f32 scale (hybrid KV row)

# Public peak specs per chip. bf16/fp8 in TF/s, HBM in GB and GB/s, ICI in
# GB/s aggregate per chip. v5e/v5p have no fp8 MXU (the compute_upcast=True
# path: fp8 storage, bf16 rate); v6e+ native fp8 at 2x bf16 (config.py
# convention). All values are vendor peaks — real kernels see 40–70%.
HW = {
    "v5e":          dict(bf16=197.0,  fp8=197.0,  hbm_gb=16,  bw=819.0,  ici=200.0),
    "v5p":          dict(bf16=459.0,  fp8=459.0,  hbm_gb=95,  bw=2765.0, ici=600.0),
    "v6e Trillium": dict(bf16=918.0,  fp8=1836.0, hbm_gb=32,  bw=1640.0, ici=448.0),
    "v7 Ironwood":  dict(bf16=2307.0, fp8=4614.0, hbm_gb=192, bw=7370.0, ici=1200.0),
}


@dataclass(frozen=True)
class ModelCosts:
    cfg: ModelConfig
    cnt: dict          # param counts by class
    byts: dict         # stored bytes by class (serving dtypes via eval_shape)
    f_fp8: float       # static fp8 FLOPs/token (MoE expert GEMMs)
    f_bf16: float      # static bf16 FLOPs/token (everything else)
    ici_tok: float     # ICI bytes/token, ep->inf upper bound
    n_csa: int
    n_hca: int
    n_swa: int

    @property
    def total_params(self):
        return sum(self.cnt.values())

    @property
    def total_bytes(self):
        return sum(self.byts.values())

    @property
    def per_expert_bytes(self):
        e = self.cfg.moe.n_routed * self.cfg.n_layers
        return self.byts["routed"] / e

    @property
    def dense_stream_bytes(self):
        # attn/mhc/router/head stream as bf16 in serving (init keeps f32
        # masters; 2 B/param is the serving rate), shared expert as stored fp8.
        return 2.0 * (self.cnt["other"] + self.cnt["head"]) + self.byts["shared"]

    def dyn(self, S: int) -> tuple[float, float]:
        """(FLOPs, HBM bytes) per decode token from context-dependent attention."""
        cs, hc_ = self.cfg.csa, self.cfg.hca
        n_blk = S // cs.m
        ne = S // hc_.m_prime
        f = (self.n_csa * (2 * 2 * cs.n_h * cs.c * (cs.topk + cs.n_win)
                           + 2 * cs.n_I_h * cs.c_I * n_blk)
             + self.n_hca * (2 * 2 * hc_.n_h * hc_.c * (ne + hc_.n_win))
             + self.n_swa * (2 * 2 * cs.n_h * cs.c * cs.n_win))
        b = (self.n_csa * ((cs.topk + cs.n_win) * ROW_BYTES + n_blk * cs.c_I * 2)
             + self.n_hca * ((ne + hc_.n_win) * ROW_BYTES)
             + self.n_swa * (cs.n_win * ROW_BYTES))
        return f, b

    def kv_seq_bytes(self, S: int) -> float:
        """Per-sequence cache bytes at context S — the *architectural* cost
        (ring-buffered SWA window). NB: model.py's init_state currently
        materializes the raw SWA cache at full s_max (580 B x S x n_layers
        ~ 25 GB/seq at 1M Flash) — that must become a ring buffer before
        1M serving; see HARDWARE_NOTES §12."""
        cs, hc_ = self.cfg.csa, self.cfg.hca
        per_csa = (S // cs.m) * (ROW_BYTES + cs.c_I * 2) + cs.n_win * ROW_BYTES
        per_hca = (S // hc_.m_prime) * ROW_BYTES + hc_.n_win * ROW_BYTES
        per_swa = cs.n_win * ROW_BYTES
        return self.n_csa * per_csa + self.n_hca * per_hca + self.n_swa * per_swa

    @property
    def act_bytes_tok(self):
        # mHC residual-stream round trips per token: per half, the fused
        # kernels read X (hc*d), read X+f, write hc*d in bf16 (§1) -> ~6*hc*d*2
        # per layer. Matters only in the memory-bound regime.
        return 6 * self.cfg.hc * self.cfg.d * 2 * self.cfg.n_layers


def model_costs(cfg: ModelConfig) -> ModelCosts:
    # `LayerParams.kind` is a static str, so capture leaf shapes as a tracing
    # side effect instead of returning the tree from eval_shape.
    info = []

    def grab(k):
        params = init_params(k, cfg)
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
            if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
                info.append((jax.tree_util.keystr(path), leaf.shape, jnp.dtype(leaf.dtype)))
        return jnp.zeros(())

    jax.eval_shape(grab, jax.random.PRNGKey(0))
    cnt = dict(routed=0, shared=0, other=0, embed=0, head=0)
    byts = dict(routed=0, shared=0, other=0, embed=0, head=0)
    for p, shape, dtype in info:
        n = math.prod(shape)
        if ".moe" in p and "shared" in p:
            k = "shared"
        elif ".moe" in p and (".w13" in p or ".w2" in p):
            k = "routed"
        elif ".embed" in p:
            k = "embed"
        elif ".head" in p:
            k = "head"
        else:
            k = "other"
        cnt[k] += n
        byts[k] += n * dtype.itemsize

    moe = cfg.moe
    act_routed = cnt["routed"] * moe.topk / moe.n_routed
    sched = layer_schedule(cfg)
    # ICI: fp8 dispatch (+scales) and bf16 combine per token per MoE layer.
    ici_tok = len(sched) * (moe.topk * (cfg.d * 1 + cfg.d // 128 * 4 + 16)
                            + moe.topk * cfg.d * 2)
    return ModelCosts(
        cfg=cfg, cnt=cnt, byts=byts,
        f_fp8=2.0 * (act_routed + cnt["shared"]),
        f_bf16=2.0 * (cnt["other"] + cnt["head"]),
        ici_tok=ici_tok,
        n_csa=sched.count("csa"), n_hca=sched.count("hca"), n_swa=sched.count("swa"),
    )


def min_chips(mc: ModelCosts, h: dict) -> int:
    return math.ceil(mc.total_bytes / (h["hbm_gb"] * 0.85e9))


def analyze(name: str, cfg: ModelConfig, contexts=(32768, 131072, 1048576)) -> None:
    """Compute ceiling: decode at the large-batch limit."""
    mc = model_costs(cfg)
    cnt = mc.cnt
    act = (cnt["routed"] * cfg.moe.topk / cfg.moe.n_routed
           + cnt["shared"] + cnt["other"] + cnt["head"])  # embed = lookup

    print(f"\n=== {name} ===")
    print(f"layers: {cfg.n_layers}  (csa={mc.n_csa} hca={mc.n_hca} swa-intro={mc.n_swa})")
    print(f"total params: {mc.total_params/1e9:.1f}B   weights on HBM: {mc.total_bytes/1e9:.1f} GB")
    print(f"  routed experts: {cnt['routed']/1e9:.1f}B  shared: {cnt['shared']/1e9:.2f}B  "
          f"attn/mhc/router: {cnt['other']/1e9:.2f}B  embed+head: {(cnt['embed']+cnt['head'])/1e9:.2f}B")
    print(f"activated/token: {act/1e9:.2f}B  -> static FLOPs/tok = {(mc.f_fp8+mc.f_bf16)/1e9:.1f} GF "
          f"(fp8 {mc.f_fp8/1e9:.1f} + bf16 {mc.f_bf16/1e9:.1f})")

    for S in contexts:
        f_dyn, b_dyn = mc.dyn(S)
        print(f"\n-- context S={S//1024}K: +dyn {f_dyn/1e9:.1f} GF/tok, "
              f"KV+indexer reads {b_dyn/1e6:.2f} MB/tok, "
              f"cache {mc.kv_seq_bytes(S)/1e9:.2f} GB/seq, "
              f"ICI <= {mc.ici_tok/1e3:.0f} KB/tok")
        for chip, h in HW.items():
            t_compute = mc.f_fp8 / (h["fp8"] * 1e12) + (mc.f_bf16 + f_dyn) / (h["bf16"] * 1e12)
            t_bw = b_dyn / (h["bw"] * 1e9)
            t_ici = mc.ici_tok / (h["ici"] * 1e9)
            t = max(t_compute, t_bw, t_ici)
            bind = {t_compute: "compute", t_bw: "HBM-bw", t_ici: "ICI"}[t]
            print(f"  {chip:13s} {1/t:>9,.0f} tok/s/chip  ({bind}-bound; "
                  f"compute {1/t_compute:,.0f}, bw {1/t_bw:,.0f}, ici {1/t_ici:,.0f}) "
                  f" min {min_chips(mc, h)} chips for weights")


def decode_memory_bound(name: str, cfg: ModelConfig,
                        contexts=(32768, 131072, 1048576),
                        batches=(1, 8, 32, 128, 512)) -> None:
    """Memory-bound decode: per-chip batch sweep, KV-capacity-aware fleet.

    Fleet per cell: n = max(weight floor, chips so that sharded weights +
    b x kv_seq fit 85% HBM), capped at n_routed (one expert per chip = max
    EP width; past that weights replicate across dp groups and this model
    stops applying). '--' = infeasible under those rules.
    """
    mc = model_costs(cfg)
    per_exp = mc.per_expert_bytes
    E, topk, L = cfg.moe.n_routed, cfg.moe.topk, cfg.n_layers

    print(f"\n=== {name}: memory-bound decode (expected expert hits) ===")
    print(f"streamed every step/chip: dense bf16 {mc.dense_stream_bytes/1e9:.2f} GB; "
          f"per routed expert hit {per_exp/1e6:.1f} MB; "
          f"mHC residual traffic {mc.act_bytes_tok/1e6:.2f} MB/tok")

    for S in contexts:
        f_dyn, b_dyn = mc.dyn(S)
        kv = mc.kv_seq_bytes(S)
        print(f"\n-- context S={S//1024}K (KV+indexer {b_dyn/1e6:.1f} MB/tok, "
              f"cache {kv/1e9:.2f} GB/seq)")
        for chip, h in HW.items():
            n_w = min_chips(mc, h)
            cells = []
            crossover = None
            for b in batches:
                room = h["hbm_gb"] * 0.85e9 - b * kv
                n = math.ceil(mc.total_bytes / room) if room > 0 else None
                if n is None or n > E:
                    cells.append(f"b={b}: {'--':>10s}")
                    continue
                n = max(n_w, n)
                hit = (E / n) * (1.0 - (1.0 - topk / E) ** (b * n))   # experts/layer/chip
                w_bytes = mc.dense_stream_bytes + hit * per_exp * L
                step_bytes = w_bytes + b * (b_dyn + mc.act_bytes_tok)
                t_bw = step_bytes / (h["bw"] * 1e9)
                t_c = b * (mc.f_fp8 / (h["fp8"] * 1e12) + (mc.f_bf16 + f_dyn) / (h["bf16"] * 1e12))
                t_ici = b * mc.ici_tok / (h["ici"] * 1e9)
                t = max(t_bw, t_c, t_ici)
                bind = ("c" if t == t_c else "i" if t == t_ici else
                        "w" if w_bytes > b * (b_dyn + mc.act_bytes_tok) else "kv")
                if crossover is None and t_c >= t_bw:
                    crossover = b
                fleet = f"x{n}" if n != n_w else ""
                cells.append(f"b={b}: {b/t:>8,.0f}/{bind}{fleet}")
            print(f"  {chip:13s} (>= x{n_w}) {'  '.join(cells)}"
                  f"   compute-bound from b~{crossover if crossover else f'>{batches[-1]}'}")
    print("\n  cells: tok/s/chip / binding term (w=weight stream, kv=KV+indexer, "
          "c=compute, i=ICI); xN = fleet grown beyond the weight floor to hold KV; "
          "-- = infeasible (b x kv_seq exceeds a chip's HBM, or fleet would "
          "exceed max EP width); b = decode sequences per chip")


if __name__ == "__main__":
    analyze("DSv4-Flash", DSV4_FLASH)
    analyze("DSv4-Pro", DSV4_PRO)
    decode_memory_bound("DSv4-Flash", DSV4_FLASH)
    decode_memory_bound("DSv4-Pro", DSV4_PRO)
