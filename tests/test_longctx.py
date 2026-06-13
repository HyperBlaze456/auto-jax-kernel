"""Long-context memory package tests (HARDWARE_NOTES §13).

Covers the three serving-memory optimizations:
  1. SWA dual-write ring cache (capacity: 2*n_win rows instead of s_max)
  2. fp8 indexer-key cache (bandwidth: halves the dominant long-context scan)
  3. coarse-to-fine paged indexer + paged gather wired into the model
     (bandwidth: scan page summaries only; gather page slabs)
"""

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsv4 import eager
from dsv4.serving import attention, indexer_scan, model, quant
from dsv4.serving.config import SMALL_MODEL, ServingTiles

jax.config.update("jax_platform_name", "cpu")


# ---------------------------------------------------------------------------
# 1. SWA ring buffer
# ---------------------------------------------------------------------------


def _build_ring(kv: quant.KVQuant, n_written: int, n_win: int) -> quant.KVQuant:
    """Dual-write unrolled ring from a flat position-indexed cache, as the
    model maintains it after ``n_written`` rows (mirrors model._ring)."""
    def ring(a):
        t = (a[:, n_written - n_win:n_written] if n_written >= n_win
             else jnp.pad(a[:, :n_written],
                          ((0, 0), (n_win - n_written, 0), (0, 0))))
        t = jnp.roll(t, n_written % n_win, axis=1)
        return jnp.concatenate([t, t], axis=1)
    return jax.tree.map(ring, kv)


@pytest.fixture()
def ring_setup():
    ks = jax.random.split(jax.random.PRNGKey(20), 6)
    B, n_h, c, r = 2, 4, 128, 64
    S_c, S_r, k, n_win = 32, 64, 8, 16
    q = jax.random.normal(ks[0], (B, 1, n_h, c), jnp.float32)
    kc = quant.quantize_kv(jax.random.normal(ks[1], (B, S_c, c), jnp.float32), r)
    swa = quant.quantize_kv(jax.random.normal(ks[2], (B, S_r, c), jnp.float32), r)
    sink = jax.random.normal(ks[3], (n_h,), jnp.float32) * 0.5
    idx = jax.random.randint(ks[4], (B, 1, k), 0, S_c).astype(jnp.int32)
    tiles = ServingTiles(attn_chunk=4, interpret=True)
    return q, kc, idx, swa, sink, n_win, r, tiles


def test_ring_swa_bit_exact_vs_flat(ring_setup):
    """For pos >= n_win-1 the ring slab is the same rows in the same order
    as the flat path — outputs must be bit-identical (the decode gold
    test's bit-exactness budget rests on this)."""
    q, kc, idx, swa, sink, n_win, r, tiles = ring_setup
    for pos_v in (n_win - 1, 37, 63):
        pos = jnp.array([[pos_v], [pos_v]], jnp.int32)
        flat = attention.sparse_mqa_gathered(
            q, kc, idx, swa, pos, sink, n_win=n_win, rope_dim=r, tiles=tiles)
        ring = _build_ring(swa, pos_v + 1, n_win)
        rng = attention.sparse_mqa_gathered(
            q, kc, idx, ring, pos, sink, n_win=n_win, rope_dim=r,
            swa_ring=True, tiles=tiles)
        np.testing.assert_array_equal(np.asarray(flat), np.asarray(rng))


def test_ring_swa_early_positions_vs_oracle(ring_setup):
    """pos < n_win-1: the ring places the valid rows at different lanes
    than the flat path (front of window is pre-sequence), so compare
    against the dense oracle instead of bit-exactness."""
    q, kc, idx, swa, sink, n_win, r, tiles = ring_setup
    for pos_v in (0, 3, n_win - 2):
        pos = jnp.array([[pos_v], [pos_v]], jnp.int32)
        ring = _build_ring(swa, pos_v + 1, n_win)
        out = attention.sparse_mqa_gathered(
            q, kc, idx, ring, pos, sink, n_win=n_win, rope_dim=r,
            swa_ring=True, tiles=tiles)
        ref = attention.serving_attn_ref(q, kc, idx, swa, pos, sink,
                                         n_win=n_win, rope_dim=r)
        assert float(jnp.abs(out.astype(jnp.float32) - ref).max()) < 2e-2


# ---------------------------------------------------------------------------
# 2. fp8 indexer keys: selection recall
# ---------------------------------------------------------------------------


def test_fp8_ki_selection_recall():
    """fp8 ki perturbs *ranking only*. On uncorrelated gaussian keys (the
    structural worst case — real indexer scores separate more) the top-k
    overlap vs bf16 keys stays high; report and bound it."""
    ks = jax.random.split(jax.random.PRNGKey(21), 3)
    B, n, n_blk, c_I, n_h, k = 2, 16, 64, 32, 4, 8
    ki = jax.random.normal(ks[0], (B, n_blk, c_I), jnp.float32)
    qI = jax.random.normal(ks[1], (B, n, n_h, c_I), jnp.float32)
    wI = jax.random.normal(ks[2], (B, n, n_h), jnp.float32)

    def scores(k_keys):
        qk = jax.nn.relu(jnp.einsum("bthc,bsc->btsh", qI, k_keys))
        s = (wI[:, :, None, :] * qk).sum(-1)
        mask = jnp.arange(n_blk)[None, :] < (jnp.arange(n)[:, None] + 32)
        return jnp.where(mask[None], s, -jnp.inf)

    top_bf16 = eager.topk_indices(
        scores(ki.astype(jnp.bfloat16).astype(jnp.float32)), k)
    top_fp8 = eager.topk_indices(
        scores(quant.dequantize_rows(quant.quantize_rows(ki))), k)
    hit = (top_bf16[..., :, None] == top_fp8[..., None, :]).any(-1)
    valid = top_bf16 >= 0
    recall = float((hit & valid).sum() / jnp.maximum(valid.sum(), 1))
    assert recall >= 0.9, f"fp8 ki recall {recall:.3f}"


def test_row_quant_roundtrip():
    x = jax.random.normal(jax.random.PRNGKey(22), (4, 16, 32), jnp.float32)
    rq = quant.quantize_rows(x)
    err = jnp.abs(quant.dequantize_rows(rq) - x).max() / jnp.abs(x).max()
    assert rq.q.dtype == jnp.float8_e4m3fn
    assert float(err) < 0.05


# ---------------------------------------------------------------------------
# 3. paged indexer + paged gather, full model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def paged_model():
    cfg = dataclasses.replace(SMALL_MODEL, csa_pages=2)
    tiles = ServingTiles(gemm_tm=16, attn_chunk=4, mhc_bn=8, interpret=True)
    params = model.init_params(jax.random.PRNGKey(7), cfg)
    return cfg, tiles, params


def test_paged_decode_matches_prefill(paged_model):
    """The gold test under paged selection: stepwise decode must reproduce
    prefill logits (same tolerance rationale as the row-exact twin)."""
    cfg, tiles, params = paged_model
    B, S = 1, 96
    toks = jax.random.randint(jax.random.PRNGKey(9), (B, 66), 0, cfg.vocab)
    n0 = 48
    _, state = model.prefill(params, toks[:, :n0], cfg,
                             model.init_state(cfg, B, S), tiles=tiles)
    for t in range(n0, 65):
        logits_dec, state = model.decode_step(params, toks[:, t], cfg,
                                              state, tiles=tiles)
        if t in (n0, 55, 63, 64):   # 55/63 cross CSA page (P*m=8) bounds
            logits_pre, sp = model.prefill(params, toks[:, :t + 1], cfg,
                                           model.init_state(cfg, B, S),
                                           tiles=tiles)
            a, b = logits_pre[:, -1], logits_dec
            rel = float(jnp.abs(a - b).max() / jnp.abs(a).max())
            assert rel < 5e-2, f"pos {t}: rel={rel}"
    # page-summary cache written by decode must match its prefill twin
    for kind, lc_d, lc_p in zip(model.layer_schedule(cfg),
                                state.caches, sp.caches):
        if kind != "csa":
            continue
        sd = quant.dequantize_rows(lc_d.kis)
        sp_ = quant.dequantize_rows(lc_p.kis)
        assert float(jnp.abs(sd - sp_).max()) < 0.3


def test_paged_decode_is_deterministic(paged_model):
    cfg, tiles, params = paged_model
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


def test_paged_prefill_logits_finite(paged_model):
    cfg, tiles, params = paged_model
    toks = jax.random.randint(jax.random.PRNGKey(10), (2, 24), 0, cfg.vocab)
    logits, state = model.prefill(params, toks, cfg,
                                  model.init_state(cfg, 2, 64), tiles=tiles)
    assert logits.shape == (2, 24, cfg.vocab)
    assert bool(jnp.isfinite(logits).all())


# ---------------------------------------------------------------------------
# 4. valid-prefix indexer scan kernel (HARDWARE_NOTES §13.5)
# ---------------------------------------------------------------------------


@pytest.fixture()
def scan_setup():
    """Standalone kernel inputs sized for multiple scan chunks (256 rows =
    2 chunks of 128) so the pl.when DMA skip and the prefix-edge partial
    chunk are both exercised."""
    ks = jax.random.split(jax.random.PRNGKey(31), 5)
    B, d, d_c, n_I_h, c_I, S_blk = 2, 64, 32, 4, 32, 256
    h_t = jax.random.normal(ks[0], (B, 1, d), jnp.float32).astype(jnp.bfloat16)
    p = SimpleNamespace(
        W_DQ=(jax.random.normal(ks[1], (d, d_c)) * d ** -0.5
              ).astype(jnp.bfloat16),
        W_IUQ=(jax.random.normal(ks[2], (d_c, n_I_h * c_I)) * d_c ** -0.5
               ).astype(jnp.bfloat16),
        W_w=(jax.random.normal(ks[3], (d, n_I_h)) * d ** -0.5
             ).astype(jnp.bfloat16),
    )
    acfg = SimpleNamespace(n_I_h=n_I_h, c_I=c_I)
    ki = quant.quantize_rows(
        jax.random.normal(ks[4], (B, S_blk, c_I), jnp.float32))
    tiles = ServingTiles(attn_chunk=4, interpret=True)
    return h_t, p, acfg, ki, tiles


def test_indexer_scan_matches_eager(scan_setup):
    """Kernel scores == eager reference at every prefix length, including
    n_valid on, off, and straddling the 128-row chunk boundary; the -inf
    sentinel pattern (topk_indices' isfinite contract) must be identical."""
    h_t, p, acfg, ki, tiles = scan_setup
    for nv in (0, 1, 64, 127, 128, 129, 255, 256):
        ref = np.asarray(model._indexer_scores_step(
            h_t, quant.dequantize_rows(ki), p, acfg, jnp.int32(nv)))
        got = np.asarray(indexer_scan.indexer_scores_decode(
            h_t, ki, p, acfg, jnp.int32(nv), tiles=tiles))
        np.testing.assert_array_equal(np.isfinite(got), np.isfinite(ref),
                                      err_msg=f"n_valid={nv}")
        fin = np.isfinite(ref)
        np.testing.assert_allclose(got[fin], ref[fin], rtol=2e-5, atol=2e-5,
                                   err_msg=f"n_valid={nv}")


def test_indexer_scan_topk_matches_eager(scan_setup):
    """Selection (what downstream actually consumes) must agree exactly."""
    h_t, p, acfg, ki, tiles = scan_setup
    for nv in (64, 129, 256):
        ref = model._indexer_scores_step(
            h_t, quant.dequantize_rows(ki), p, acfg, jnp.int32(nv))
        got = indexer_scan.indexer_scores_decode(
            h_t, ki, p, acfg, jnp.int32(nv), tiles=tiles)
        np.testing.assert_array_equal(
            np.asarray(eager.topk_indices(got, 8)),
            np.asarray(eager.topk_indices(ref, 8)), err_msg=f"n_valid={nv}")


def test_indexer_scan_rejects_misaligned_cache(scan_setup):
    h_t, p, acfg, _, tiles = scan_setup
    ki = quant.quantize_rows(jnp.zeros((2, 192, acfg.c_I)))
    with pytest.raises(ValueError, match="multiple of the scan chunk"):
        indexer_scan.indexer_scores_decode(h_t, ki, p, acfg, jnp.int32(0),
                                           tiles=tiles)


def test_decode_independent_of_allocation():
    """The valid-prefix property, end to end: the same tokens decoded under
    a small (1-chunk) and a large (multi-chunk) cache allocation must give
    bit-identical logits — scores, selection, and the gather can depend on
    position only, never on s_max."""
    cfg = SMALL_MODEL
    tiles = ServingTiles(gemm_tm=16, attn_chunk=4, mhc_bn=8, interpret=True)
    params = model.init_params(jax.random.PRNGKey(7), cfg)
    toks = jax.random.randint(jax.random.PRNGKey(13), (1, 52), 0, cfg.vocab)
    outs = []
    for s_max in (96, 1024):   # S_blk = 24 (full-dim chunk) vs 256 (2x128)
        _, state = model.prefill(params, toks[:, :48], cfg,
                                 model.init_state(cfg, 1, s_max),
                                 tiles=tiles)
        step_logits = []
        for t in range(48, 52):
            logits, state = model.decode_step(params, toks[:, t], cfg,
                                              state, tiles=tiles)
            step_logits.append(np.asarray(logits))
        outs.append(np.stack(step_logits))
    np.testing.assert_array_equal(outs[0], outs[1])


# ---------------------------------------------------------------------------
# 5. exact coarse-to-fine: envelopes + rescan (HARDWARE_NOTES §13.7)
# ---------------------------------------------------------------------------


def test_envelope_bound_quantization_is_sound():
    """Directed-rounding fp8 (quantize_rows_bound): the plain f32 dequant
    must bound the original values elementwise, whichever way e4m3
    rounded — across a wide dynamic range so subnormal payloads and the
    zero-crossing ulp step are exercised too. Also tight: within one ulp
    (12.5%) + the row-scale subnormal step of the true value."""
    ks = jax.random.split(jax.random.PRNGKey(40), 2)
    x = jax.random.normal(ks[0], (4, 64, 32), jnp.float32)
    x = x * (10.0 ** jax.random.uniform(ks[1], (4, 64, 1),
                                        minval=-6.0, maxval=2.0))
    hi_q = quant.quantize_rows_bound(x, upper=True)
    lo_q = quant.quantize_rows_bound(x, upper=False)
    hi, lo = quant.dequantize_rows(hi_q), quant.dequantize_rows(lo_q)
    assert bool((hi >= x).all()) and bool((lo <= x).all())
    slack = 0.125 * jnp.abs(x) + hi_q.scale * 2.0 ** -8
    assert bool((hi - x <= slack).all()) and bool((x - lo <= slack).all())


@pytest.fixture()
def exact_setup():
    """Random fp8 ki cache + quantized max/min page envelopes (P=4,
    16 pages), plus the query-side params the kernels consume."""
    ks = jax.random.split(jax.random.PRNGKey(50), 5)
    B, d, d_c, n_I_h, c_I = 2, 64, 32, 4, 32
    S_blk, P = 64, 4
    h_t = jax.random.normal(ks[0], (B, 1, d), jnp.float32).astype(jnp.bfloat16)
    p = SimpleNamespace(
        W_DQ=(jax.random.normal(ks[1], (d, d_c)) * d ** -0.5
              ).astype(jnp.bfloat16),
        W_IUQ=(jax.random.normal(ks[2], (d_c, n_I_h * c_I)) * d_c ** -0.5
               ).astype(jnp.bfloat16),
        W_w=(jax.random.normal(ks[3], (d, n_I_h)) * d ** -0.5
             ).astype(jnp.bfloat16),
    )
    acfg = SimpleNamespace(n_I_h=n_I_h, c_I=c_I)
    ki = quant.quantize_rows(
        jax.random.normal(ks[4], (B, S_blk, c_I), jnp.float32))
    pg = quant.dequantize_rows(ki).reshape(B, S_blk // P, P, c_I)
    hi = quant.quantize_rows_bound(pg.max(axis=2), upper=True)
    lo = quant.quantize_rows_bound(pg.min(axis=2), upper=False)
    tiles = ServingTiles(attn_chunk=4, interpret=True)
    return h_t, p, acfg, ki, hi, lo, P, tiles


def test_envelope_ub_bounds_row_scores(exact_setup):
    """Soundness invariant the exactness rests on: the kernel's UB(page)
    must dominate the exact f32 score of every row in that page."""
    h_t, p, acfg, ki, hi, lo, P, tiles = exact_setup
    B, S_blk = ki.q.shape[0], ki.q.shape[1]
    n_pg = S_blk // P
    ub = np.asarray(indexer_scan.indexer_ub_scores_decode(
        h_t, hi, lo, p, acfg, jnp.int32(n_pg), tiles=tiles))
    sc = np.asarray(model._indexer_scores_step(
        h_t, quant.dequantize_rows(ki), p, acfg, jnp.int32(S_blk)))
    pg_max = sc.reshape(B, 1, n_pg, P).max(-1)
    assert (ub + 1e-6 >= pg_max).all(), float((pg_max - ub).max())


def _exact_pipeline(h_t, p, acfg, ki, hi, lo, P, R, k, nv, tiles):
    """Run the §13.7 decode selection pieces; returns (rows, certificate)."""
    nv_pages = nv // P
    ub = indexer_scan.indexer_ub_scores_decode(
        h_t, hi, lo, p, acfg, jnp.int32(nv_pages), tiles=tiles)
    top_pg = eager.topk_indices(ub, R)
    sc, rows = model._rescan_rows(h_t, ki, p, acfg, top_pg, P, jnp.int32(nv))
    got = model._topk_rows(sc, rows, k)
    cert = model._exact_certificate(ub, top_pg, sc, k)
    return got, cert


def test_exact_selection_matches_row_exact_full_budget(exact_setup):
    """With the rescan budget covering every page, the pipeline (UB scan →
    top-R pages → row rescan → top-k) must reproduce the row-exact f32
    top-k over completed pages unconditionally, at every prefix length —
    including page-fraction and empty prefixes. (Validates the plumbing:
    UB valid-prefix masking, gather indexing, candidate top-k mapping.)"""
    h_t, p, acfg, ki, hi, lo, P, tiles = exact_setup
    S_blk = ki.q.shape[1]
    k, R = 8, S_blk // P                          # rescan everything
    for nv in (0, 3, 5, 17, 32, 63, 64):
        ref = eager.topk_indices(model._indexer_scores_step(
            h_t, quant.dequantize_rows(ki), p, acfg,
            jnp.int32((nv // P) * P)), k)
        got, cert = _exact_pipeline(h_t, p, acfg, ki, hi, lo, P, R, k, nv,
                                    tiles)
        assert bool(cert.all()), f"n_valid={nv}"
        np.testing.assert_array_equal(
            np.sort(np.asarray(got), axis=-1),
            np.sort(np.asarray(ref), axis=-1), err_msg=f"n_valid={nv}")


def test_exact_selection_certificate(exact_setup):
    """The budgeted regime (R = 2*topk/P, the serving default). On iid
    gaussian keys the top-k spreads across ~k pages — the structural
    worst case — so the certificate may fail; what §13.7 promises is the
    *implication*: certificate ⇒ selection == row-exact. On page-
    clustered keys (the realistic regime the default budget targets:
    indexer keys are temporally correlated) the certificate must hold
    and the selection must be row-exact."""
    h_t, p, acfg, ki, hi, lo, P, tiles = exact_setup
    S_blk = ki.q.shape[1]
    k, R = 8, 4                                   # 2*topk/P
    # (a) implication on the iid worst case, all prefix lengths
    for nv in (17, 32, 64):
        ref = eager.topk_indices(model._indexer_scores_step(
            h_t, quant.dequantize_rows(ki), p, acfg,
            jnp.int32((nv // P) * P)), k)
        got, cert = _exact_pipeline(h_t, p, acfg, ki, hi, lo, P, R, k, nv,
                                    tiles)
        eq = (np.sort(np.asarray(got), -1)
              == np.sort(np.asarray(ref), -1)).all(-1)   # [B, 1]
        assert (~np.asarray(cert) | eq).all(), f"n_valid={nv}"
    # (b) clustered keys: per-page centroid + small noise → tight
    # envelopes, top rows share pages → certificate holds, selection
    # exact. Needs R = 3*topk/P here: coordinatewise envelope bounds are
    # pessimistic against dense queries (UB inflation ~ SUM_c |q_c| *
    # envelope width), so serving should size csa_rescan by monitoring
    # _exact_certificate, not assume the 2*topk/P default.
    ks = jax.random.split(jax.random.PRNGKey(60), 2)
    B, c_I, n_pg = ki.q.shape[0], ki.q.shape[2], S_blk // P
    cent = 2.0 * jax.random.normal(ks[0], (B, n_pg, 1, c_I), jnp.float32)
    noise = 0.05 * jax.random.normal(ks[1], (B, n_pg, P, c_I), jnp.float32)
    ki_c = quant.quantize_rows((cent + noise).reshape(B, S_blk, c_I))
    pg = quant.dequantize_rows(ki_c).reshape(B, n_pg, P, c_I)
    hi_c = quant.quantize_rows_bound(pg.max(axis=2), upper=True)
    lo_c = quant.quantize_rows_bound(pg.min(axis=2), upper=False)
    for nv in (32, 64):
        ref = eager.topk_indices(model._indexer_scores_step(
            h_t, quant.dequantize_rows(ki_c), p, acfg,
            jnp.int32((nv // P) * P)), k)
        got, cert = _exact_pipeline(h_t, p, acfg, ki_c, hi_c, lo_c, P, 6,
                                    k, nv, tiles)
        assert bool(cert.all()), f"clustered n_valid={nv}"
        np.testing.assert_array_equal(
            np.sort(np.asarray(got), axis=-1),
            np.sort(np.asarray(ref), axis=-1),
            err_msg=f"clustered n_valid={nv}")


@pytest.fixture(scope="module")
def exact_model():
    cfg = dataclasses.replace(SMALL_MODEL, csa_pages=2, csa_pages_exact=True)
    tiles = ServingTiles(gemm_tm=16, attn_chunk=4, mhc_bn=8, interpret=True)
    params = model.init_params(jax.random.PRNGKey(7), cfg)
    return cfg, tiles, params


def test_exact_decode_matches_prefill(exact_model):
    """Gold test under exact coarse-to-fine selection: stepwise decode
    must reproduce prefill logits (both sides select the row-exact f32
    top-k over completed pages — one shared contract)."""
    cfg, tiles, params = exact_model
    B, S = 1, 96
    toks = jax.random.randint(jax.random.PRNGKey(9), (B, 66), 0, cfg.vocab)
    n0 = 48
    _, state = model.prefill(params, toks[:, :n0], cfg,
                             model.init_state(cfg, B, S), tiles=tiles)
    for t in range(n0, 65):
        logits_dec, state = model.decode_step(params, toks[:, t], cfg,
                                              state, tiles=tiles)
        if t in (n0, 55, 63, 64):
            logits_pre, sp = model.prefill(params, toks[:, :t + 1], cfg,
                                           model.init_state(cfg, B, S),
                                           tiles=tiles)
            a, b = logits_pre[:, -1], logits_dec
            rel = float(jnp.abs(a - b).max() / jnp.abs(a).max())
            assert rel < 5e-2, f"pos {t}: rel={rel}"
    # envelope caches written by decode must match their prefill twins
    for kind, lc_d, lc_p in zip(model.layer_schedule(cfg),
                                state.caches, sp.caches):
        if kind != "csa":
            continue
        for fld in ("kis", "kis_lo"):
            sd = quant.dequantize_rows(getattr(lc_d, fld))
            sp_ = quant.dequantize_rows(getattr(lc_p, fld))
            assert float(jnp.abs(sd - sp_).max()) < 0.3, fld


def test_exact_decode_is_deterministic(exact_model):
    cfg, tiles, params = exact_model
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


# ---------------------------------------------------------------------------
# 6. q-block batched prefill gather (HARDWARE_NOTES §13.8)
# ---------------------------------------------------------------------------


def test_blocked_union_builder():
    """The dedup/compact preamble: union must be the distinct non-negative
    selected blocks (front-packed, -1 padded), D their count, and
    membership[t,j] iff union[j] is in token t's top-k."""
    from dsv4.serving.attention_blocked import _build_union
    B, t_pad, k, bq = 1, 8, 4, 4          # 2 blocks of 4 tokens
    topk = jnp.array([[[0, 2, 2, -1], [2, 5, 0, -1], [1, 1, 1, 1],
                       [0, 3, 5, 7],
                       [9, 9, -1, -1], [8, 9, 10, 11], [8, -1, -1, -1],
                       [12, 13, 8, 9]]], jnp.int32)
    k_union = bq * k
    union, d, memb = _build_union(topk, bq, k_union)
    union, d, memb = map(np.asarray, (union, d, memb))
    # block 0 distinct = {0,1,2,3,5,7} -> 6 ; block 1 = {8,9,10,11,12,13} -> 6
    assert list(d[0]) == [6, 6]
    for nb in range(2):
        present = set(int(x) for x in union[0, nb] if x >= 0)
        ref = set(int(x) for x in topk[0, nb * bq:(nb + 1) * bq].reshape(-1)
                  if x >= 0)
        assert present == ref
        for t in range(bq):
            tk = set(int(x) for x in topk[0, nb * bq + t] if x >= 0)
            for j in range(k_union):
                u = int(union[0, nb, j])
                assert bool(memb[0, nb, t, j]) == (u >= 0 and u in tk)


def test_blocked_matches_per_token():
    """Block-batched prefill gather == the per-token kernel (same rows,
    same per-token masking; differs only by flash reassociation)."""
    from dsv4.serving import attention
    from dsv4.serving.attention_blocked import sparse_mqa_gathered_blocked
    ks = jax.random.split(jax.random.PRNGKey(70), 5)
    B, T, n_h, c, r = 2, 12, 4, 128, 64
    S_c, S_r, k, n_win = 32, 64, 8, 16
    q = jax.random.normal(ks[0], (B, T, n_h, c), jnp.float32)
    kc = quant.quantize_kv(jax.random.normal(ks[1], (B, S_c, c), jnp.float32), r)
    swa = quant.quantize_kv(jax.random.normal(ks[2], (B, S_r, c), jnp.float32), r)
    sink = jax.random.normal(ks[3], (n_h,), jnp.float32) * 0.5
    # realistic top-k: distinct indices per token (as eager.topk_indices
    # emits), causal so block s is selectable only for s < i//2 + 1.
    sc = jax.random.normal(ks[4], (B, T, S_c), jnp.float32)
    causal = jnp.arange(S_c)[None, None, :] < (jnp.arange(T)[None, :, None] // 2 + 1)
    topk = eager.topk_indices(jnp.where(causal, sc, -jnp.inf), k)
    pos = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32), (B, T))
    tiles = ServingTiles(attn_chunk=4, interpret=True)
    ref = attention.sparse_mqa_gathered(q, kc, topk, swa, pos, sink,
                                        n_win=n_win, rope_dim=r, tiles=tiles)
    for bq in (1, 4, 6):
        got = sparse_mqa_gathered_blocked(q, kc, topk, swa, sink, n_win=n_win,
                                          rope_dim=r, bq=bq, tiles=tiles)
        assert got.shape == ref.shape
        err = float(jnp.abs(got.astype(jnp.float32)
                            - ref.astype(jnp.float32)).max())
        assert err < 2e-2, f"bq={bq}: max err {err}"


def test_blocked_prefill_full_model_runs_deterministic():
    """The wired blocked path (attn_bq>1) on the full model: finite logits,
    correct shape, and run-to-run determinism. (Numeric agreement with the
    per-token path is the standalone blocked≡per-token test; blocked is an
    opt-in prefill mode, not the decode bit-twin — see §13.8.)"""
    cfg = SMALL_MODEL
    params = model.init_params(jax.random.PRNGKey(7), cfg)
    toks = jax.random.randint(jax.random.PRNGKey(5), (2, 40), 0, cfg.vocab)
    for bq in (4, 8):
        tiles = ServingTiles(gemm_tm=16, attn_chunk=4, mhc_bn=8,
                             attn_bq=bq, interpret=True)
        l1, _ = model.prefill(params, toks, cfg,
                              model.init_state(cfg, 2, 64), tiles=tiles)
        l2, _ = model.prefill(params, toks, cfg,
                              model.init_state(cfg, 2, 64), tiles=tiles)
        assert l1.shape == (2, 40, cfg.vocab)
        assert bool(jnp.isfinite(l1).all()), f"bq={bq}"
        np.testing.assert_array_equal(np.asarray(l1), np.asarray(l2))
