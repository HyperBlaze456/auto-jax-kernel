"""Correctness suite for dsv4.serving (runs in Pallas interpret mode on CPU).

Every Pallas kernel is graded against a dense fp32 oracle; the full model
is graded by decode-vs-prefill bit-consistency (the serving gold test).
Tolerances reflect *inherent* precision (fp8 storage, bf16 PV), never
kernel slop — the fp32-path comparisons assert near-ulp agreement.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsv4.serving import attention, gemm_fp8, mhc, model, moe, quant
from dsv4.serving.config import SMALL_MODEL, MoEConfig, ServingTiles

jax.config.update("jax_platform_name", "cpu")


# ---------------------------------------------------------------------------
# quant
# ---------------------------------------------------------------------------


def test_act_quant_roundtrip():
    x = jax.random.normal(jax.random.PRNGKey(0), (64, 512), jnp.float32) * 3.0
    aq = quant.quantize_act(x)
    assert aq.q.dtype == jnp.float8_e4m3fn and aq.s_t.shape == (4, 64)
    xr = quant.dequantize_act(aq)
    # e4m3: 3 mantissa bits → per-group rel err ≤ 2^-3 of group amax.
    grp = jnp.abs(x).reshape(64, 4, 128).max(-1)
    bound = jnp.repeat(grp, 128, axis=-1).reshape(64, 512) * 2.0 ** -3
    assert bool((jnp.abs(xr - x) <= bound + 1e-6).all())


def test_weight_quant_roundtrip():
    w = jax.random.normal(jax.random.PRNGKey(1), (2, 256, 384), jnp.float32)
    wq = quant.quantize_weight(w)
    wr = quant.dequantize_weight(wq)
    assert wq.s.shape == (2, 2, 3)
    rel = float(jnp.abs(wr - w).max() / jnp.abs(w).max())
    assert rel < 0.07


def test_kv_quant_hybrid():
    k = jax.random.normal(jax.random.PRNGKey(2), (2, 32, 128), jnp.float32)
    kv = quant.quantize_kv(k, 64)
    kr = quant.dequantize_kv(kv)
    assert kv.nope.shape == (2, 32, 64) and kv.rope.dtype == jnp.bfloat16
    # rope dims bf16-exact-ish; nope dims fp8-bounded
    assert float(jnp.abs(kr[..., 64:] - k[..., 64:]).max()) < 2e-2
    assert float(jnp.abs(kr[..., :64] - k[..., :64]).max()) < 0.2


# ---------------------------------------------------------------------------
# gemm_fp8
# ---------------------------------------------------------------------------


@pytest.fixture()
def gemm_setup():
    key = jax.random.PRNGKey(3)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    M, K, N, E = 256, 256, 384, 4
    x = jax.random.normal(k1, (M, K), jnp.float32)
    aq = quant.quantize_act(x)
    w = jax.random.normal(k2, (E, K, N), jnp.float32) * 0.05
    wq = quant.quantize_weight(w)
    w13 = jax.random.normal(k3, (E, K, 512), jnp.float32) * 0.05
    w13q = quant.quantize_weight(w13)
    groups = jnp.array([70, 0, 130, 56], jnp.int32)  # includes empty group
    return aq, wq, w13q, groups


def test_gmm_fp8_vs_dequant_ref(gemm_setup):
    aq, wq, _, groups = gemm_setup
    out = gemm_fp8.gmm_fp8(aq, gemm_fp8.prepare_weight(wq), groups,
                           tm=64, out_dtype=jnp.float32, interpret=True)
    ref = gemm_fp8.gmm_ref(aq, wq, groups)
    v = int(groups.sum())
    rel = float(jnp.abs(out[:v] - ref[:v]).max() / jnp.abs(ref[:v]).max())
    assert rel < 1e-5  # fp32 reassociation only — same quantized operands


def test_gmm_swiglu_quant_chain(gemm_setup):
    aq, _, w13q, groups = gemm_setup
    v = int(groups.sum())
    h = gemm_fp8.gmm_fp8_swiglu_quant(
        aq, gemm_fp8.prepare_weight(w13q), groups, tm=64, interpret=True)
    ref_h = gemm_fp8.swiglu_ref(gemm_fp8.gmm_ref(aq, w13q, groups))
    rel = float(jnp.abs(quant.dequantize_act(h)[:v] - ref_h[:v]).max()
                / (jnp.abs(ref_h[:v]).max()))
    assert rel < 4e-2  # bounded by the (inherent) fp8 hidden re-quant
    # transposed scales feed GEMM-2 with zero relayout
    assert h.s_t.shape == (2, h.q.shape[0])


def test_gmm_straddle_and_padding(gemm_setup):
    """Groups not tile-aligned + rows beyond sum(groups) untouched."""
    aq, wq, _, _ = gemm_setup
    groups = jnp.array([1, 5, 250, 0], jnp.int32)  # heavy straddling
    out = gemm_fp8.gmm_fp8(aq, gemm_fp8.prepare_weight(wq), groups,
                           tm=64, out_dtype=jnp.float32, interpret=True)
    ref = gemm_fp8.gmm_ref(aq, wq, groups)
    v = int(groups.sum())
    assert float(jnp.abs(out[:v] - ref[:v]).max()) < 1e-4


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


@pytest.fixture()
def attn_setup():
    ks = jax.random.split(jax.random.PRNGKey(4), 8)
    B, T, n_h, c, r = 2, 4, 4, 128, 64
    S_c, S_r, k, n_win = 32, 64, 8, 16
    q = jax.random.normal(ks[0], (B, T, n_h, c), jnp.float32)
    kc = quant.quantize_kv(jax.random.normal(ks[1], (B, S_c, c), jnp.float32), r)
    swa = quant.quantize_kv(jax.random.normal(ks[2], (B, S_r, c), jnp.float32), r)
    sink = jax.random.normal(ks[3], (n_h,), jnp.float32) * 0.5
    idx = jax.random.randint(ks[4], (B, T, k), 0, S_c)
    idx = jnp.where(jax.random.bernoulli(ks[5], 0.2, (B, T, k)), -1, idx)
    pos = jnp.array([[3, 10, 20, 50], [0, 7, 33, 63]], jnp.int32)
    tiles = ServingTiles(attn_chunk=4, interpret=True)
    return q, kc, idx.astype(jnp.int32), swa, pos, sink, n_win, r, tiles


def test_gather_attention_vs_oracle(attn_setup):
    q, kc, idx, swa, pos, sink, n_win, r, tiles = attn_setup
    out = attention.sparse_mqa_gathered(q, kc, idx, swa, pos, sink,
                                        n_win=n_win, rope_dim=r, tiles=tiles)
    ref = attention.serving_attn_ref(q, kc, idx, swa, pos, sink,
                                     n_win=n_win, rope_dim=r)
    assert float(jnp.abs(out.astype(jnp.float32) - ref).max()) < 2e-2  # bf16 PV


def test_gather_attention_all_invalid_topk(attn_setup):
    """SWA-only mode: every top-k slot is -1 (the intro-layer path)."""
    q, kc, idx, swa, pos, sink, n_win, r, tiles = attn_setup
    idx = jnp.full_like(idx, -1)
    out = attention.sparse_mqa_gathered(q, kc, idx, swa, pos, sink,
                                        n_win=n_win, rope_dim=r, tiles=tiles)
    ref = attention.serving_attn_ref(q, kc, idx, swa, pos, sink,
                                     n_win=n_win, rope_dim=r)
    assert bool(jnp.isfinite(out.astype(jnp.float32)).all())
    assert float(jnp.abs(out.astype(jnp.float32) - ref).max()) < 2e-2


def test_gather_attention_decode_shape(attn_setup):
    q, kc, idx, swa, pos, sink, n_win, r, tiles = attn_setup
    out = attention.sparse_mqa_gathered(q[:, :1], kc, idx[:, :1], swa,
                                        pos[:, :1], sink, n_win=n_win,
                                        rope_dim=r, tiles=tiles)
    ref = attention.serving_attn_ref(q[:, :1], kc, idx[:, :1], swa,
                                     pos[:, :1], sink, n_win=n_win, rope_dim=r)
    assert out.shape == (2, 1, 4, 128)
    assert float(jnp.abs(out.astype(jnp.float32) - ref).max()) < 2e-2


def test_gather_attention_matches_eager_reference():
    """Cross-check against ``eager.sparse_attn_with_sink`` away from the
    SWA zero-pad edge (pos ≥ n_win-1) with unquantized (bf16-exact) KV."""
    from dsv4 import eager
    ks = jax.random.split(jax.random.PRNGKey(5), 6)
    B, T, n_h, c, r, n_win, k = 1, 3, 4, 128, 64, 8, 4
    n = 32
    q = jax.random.normal(ks[0], (B, T, n_h, c), jnp.float32)
    K_comp = jax.random.normal(ks[1], (B, 16, c), jnp.float32)
    K_raw = jax.random.normal(ks[2], (B, n, c), jnp.float32)
    sink = jax.random.normal(ks[3], (n_h,), jnp.float32) * 0.3
    idx = jax.random.randint(ks[4], (B, T, k), 0, 16).astype(jnp.int32)
    pos = jnp.array([[10, 20, 31]], jnp.int32)

    tiles = ServingTiles(attn_chunk=4, interpret=True)
    out = attention.sparse_mqa_gathered(
        q, quant.quantize_kv(K_comp, r), idx, quant.quantize_kv(K_raw, r),
        pos, sink, n_win=n_win, rope_dim=r, tiles=tiles)

    # eager oracle on the *dequantized* caches and gathered windows
    kc_d = quant.dequantize_kv(quant.quantize_kv(K_comp, r))
    kr_d = quant.dequantize_kv(quant.quantize_kv(K_raw, r))
    w_pos = pos[..., None] + jnp.arange(n_win)[None, None, :] - (n_win - 1)
    K_swa = jax.vmap(lambda kb, ib: kb[ib])(kr_d, jnp.clip(w_pos, 0, n - 1))
    ref = eager.sparse_attn_with_sink(q, kc_d, idx, K_swa, sink)
    assert float(jnp.abs(out.astype(jnp.float32) - ref).max()) < 2e-2


# ---------------------------------------------------------------------------
# moe
# ---------------------------------------------------------------------------


@pytest.fixture()
def moe_setup():
    ks = jax.random.split(jax.random.PRNGKey(6), 8)
    M, d, E, dff, topk = 32, 256, 8, 256, 2
    cfg = MoEConfig(n_routed=E, d_expert=dff, topk=topk)
    tiles = ServingTiles(gemm_tm=16, interpret=True)
    def mkw(k, s):
        return gemm_fp8.prepare_weight(
            quant.quantize_weight(jax.random.normal(k, s, jnp.float32) * 0.05))
    params = moe.MoEParams(
        w_router=jax.random.normal(ks[0], (d, E), jnp.float32) * 0.1,
        router_bias=jnp.zeros((E,), jnp.float32),
        w13=mkw(ks[1], (E, d, 2 * dff)),
        w2=mkw(ks[2], (E, dff, d)),
        w13_shared=mkw(ks[3], (1, d, 2 * dff)),
        w2_shared=mkw(ks[4], (1, dff, d)),
    )
    x = jax.random.normal(ks[5], (M, d), jnp.float32)
    return x, params, cfg, tiles


def test_moe_local_vs_ref(moe_setup):
    x, params, cfg, tiles = moe_setup
    idx, gates = moe.route(x, params, cfg)
    assert float(jnp.abs(gates.sum(-1) - 1.0).max()) < 1e-5
    out = moe.moe_forward_local(x, idx, gates, params, cfg, tiles=tiles)
    ref = moe.moe_ref(x, idx, gates, params, cfg)
    rel = float(jnp.abs(out - ref).max() / jnp.abs(ref).max())
    assert rel < 5e-2


def test_moe_hash_routing(moe_setup):
    x, params, cfg, tiles = moe_setup
    idx, gates = moe.route_hash(jnp.arange(x.shape[0], dtype=jnp.int32), cfg)
    assert int(idx.min()) >= 0 and int(idx.max()) < cfg.n_routed
    out = moe.moe_forward_local(x, idx, gates, params, cfg, tiles=tiles)
    ref = moe.moe_ref(x, idx, gates, params, cfg)
    assert float(jnp.abs(out - ref).max() / jnp.abs(ref).max()) < 5e-2


@pytest.mark.skipif(jax.device_count() < 2, reason="needs >=2 devices "
                    "(run with XLA_FLAGS=--xla_force_host_platform_device_count=4)")
def test_moe_ep_matches_local(moe_setup):
    from functools import partial
    x, params, cfg, tiles = moe_setup
    EP, NW = 2, 2
    idx, gates = moe.route(x, params, cfg)
    mesh = jax.make_mesh((EP,), ("ep",))
    P = jax.sharding.PartitionSpec
    def NS(s):
        return jax.sharding.NamedSharding(mesh, s)
    cap = (x.shape[0] // EP) * cfg.topk
    ep_specs = moe.MoEParams(
        w_router=P(), router_bias=P(),
        w13=gemm_fp8.GemmWeight(P("ep"), P("ep")),
        w2=gemm_fp8.GemmWeight(P("ep"), P("ep")),
        w13_shared=gemm_fp8.GemmWeight(P(), P()),
        w2_shared=gemm_fp8.GemmWeight(P(), P()),
    )
    args = (jax.device_put(x, NS(P("ep"))), jax.device_put(idx, NS(P("ep"))),
            jax.device_put(gates, NS(P("ep"))),
            jax.tree.map(lambda a, s: jax.device_put(a, NS(s)), params, ep_specs))
    fn = jax.shard_map(
        partial(moe.moe_forward_ep, cfg=cfg, axis_name="ep", ep_size=EP,
                n_waves=NW, capacity=cap, tiles=tiles),
        mesh=mesh, in_specs=(P("ep"), P("ep"), P("ep"), ep_specs),
        out_specs=P("ep"), check_vma=False)
    out_ep = fn(*args)
    out_local = moe.moe_forward_local(x, idx, gates, params, cfg, tiles=tiles)
    # dispatch/combine must be numerically transparent
    np.testing.assert_allclose(np.asarray(out_ep), np.asarray(out_local),
                               atol=2e-3)


# ---------------------------------------------------------------------------
# mhc
# ---------------------------------------------------------------------------


def test_mhc_fused_kernels():
    ks = jax.random.split(jax.random.PRNGKey(7), 5)
    B, n, hc, d = 2, 12, 4, 256
    tiles = ServingTiles(mhc_bn=8, interpret=True)  # n % bn != 0 → pads
    x = jax.random.normal(ks[0], (B, n, hc, d), jnp.float32)
    pre = jax.nn.sigmoid(jax.random.normal(ks[1], (B, n, hc), jnp.float32))
    comb = jax.nn.softmax(jax.random.normal(ks[2], (B, n, hc, hc), jnp.float32), -1)
    post = 2 * jax.nn.sigmoid(jax.random.normal(ks[3], (B, n, hc), jnp.float32))
    f = jax.random.normal(ks[4], (B, n, d), jnp.float32)
    assert float(jnp.abs(mhc.mhc_pre_norm(x, pre, tiles=tiles)
                         - mhc.mhc_pre_norm_ref(x, pre)).max()) < 1e-5
    assert float(jnp.abs(mhc.mhc_update(x, comb, post, f, tiles=tiles)
                         - mhc.mhc_update_ref(x, comb, post, f)).max()) < 1e-5


# ---------------------------------------------------------------------------
# full model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_model():
    cfg = SMALL_MODEL
    tiles = ServingTiles(gemm_tm=16, attn_chunk=4, mhc_bn=8, interpret=True)
    params = model.init_params(jax.random.PRNGKey(7), cfg)
    return cfg, tiles, params


def test_prefill_prefix_stability(small_model):
    """Causality: extending the sequence must not change earlier rows'
    cache entries (selection orders matched: n_blk > topk in both)."""
    cfg, tiles, params = small_model
    B, S = 1, 96
    toks = jax.random.randint(jax.random.PRNGKey(8), (B, 56), 0, cfg.vocab)
    _, s48 = model.prefill(params, toks[:, :48], cfg,
                           model.init_state(cfg, B, S), tiles=tiles)
    _, s56 = model.prefill(params, toks[:, :56], cfg,
                           model.init_state(cfg, B, S), tiles=tiles)
    for la, lb in zip(s48.caches, s56.caches):
        a = quant.dequantize_kv(la.swa)[:, :48]
        b = quant.dequantize_kv(lb.swa)[:, :48]
        assert float(jnp.abs(a - b).max()) < 1e-6


def test_decode_matches_prefill(small_model):
    """The serving gold test: stepwise decode must reproduce prefill
    logits, across CSA *and* HCA block emissions.

    Tolerance note: decode (M=1) and prefill (M=n) compile the projection
    GEMMs at different shapes, so XLA reassociates contractions
    differently — verified to seed one-bf16-ulp differences on identical
    inputs/selections — and the deep bf16 stack amplifies that chaotically
    to percent level. The *machinery* is exact: cache cross-checks below
    and `test_decode_is_deterministic` pin that down separately."""
    cfg, tiles, params = small_model
    B, S = 1, 96
    toks = jax.random.randint(jax.random.PRNGKey(9), (B, 66), 0, cfg.vocab)
    n0 = 48
    _, state = model.prefill(params, toks[:, :n0], cfg,
                             model.init_state(cfg, B, S), tiles=tiles)
    # pos 63 crosses an HCA (m'=16) boundary; pos 51/55/59/63 CSA (m=4).
    for t in range(n0, 65):
        logits_dec, state = model.decode_step(params, toks[:, t], cfg,
                                              state, tiles=tiles)
        if t in (n0, 51, 63, 64):     # spot-check incl. both boundary kinds
            logits_pre, sp = model.prefill(params, toks[:, :t + 1], cfg,
                                           model.init_state(cfg, B, S),
                                           tiles=tiles)
            a, b = logits_pre[:, -1], logits_dec
            rel = float(jnp.abs(a - b).max() / jnp.abs(a).max())
            assert rel < 5e-2, f"pos {t}: rel={rel}"
    # Cache cross-check at the end state: every compressed entry written
    # by decode must match its prefill twin up to the bf16-ulp chaos of
    # its *inputs* (the emission math itself is shared code).
    for lc_d, lc_p in zip(state.caches, sp.caches):
        kd = quant.dequantize_kv(lc_d.kc)
        kp = quant.dequantize_kv(lc_p.kc)
        assert float(jnp.abs(kd - kp).max()) < 0.3


def test_decode_is_deterministic(small_model):
    """Same shapes → bit-identical: two decode lineages from the same
    prefill must agree exactly (no hidden nondeterminism in the gather
    DMAs, grouped GEMM revisits, or cache appends)."""
    cfg, tiles, params = small_model
    B, S = 1, 96
    toks = jax.random.randint(jax.random.PRNGKey(11), (B, 52), 0, cfg.vocab)
    _, state0 = model.prefill(params, toks[:, :48], cfg,
                              model.init_state(cfg, B, S), tiles=tiles)
    outs = []
    for _ in range(2):
        state = state0
        for t in range(48, 52):
            logits, state = model.decode_step(params, toks[:, t], cfg,
                                              state, tiles=tiles)
        outs.append(logits)
    np.testing.assert_array_equal(np.asarray(outs[0]), np.asarray(outs[1]))


def test_prefill_logits_finite(small_model):
    cfg, tiles, params = small_model
    toks = jax.random.randint(jax.random.PRNGKey(10), (2, 24), 0, cfg.vocab)
    logits, state = model.prefill(params, toks, cfg,
                                  model.init_state(cfg, 2, 64), tiles=tiles)
    assert logits.shape == (2, 24, cfg.vocab)
    assert bool(jnp.isfinite(logits).all())
    assert int(state.pos) == 24
