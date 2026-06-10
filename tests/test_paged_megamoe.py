"""Tests for page-aligned gather attention + EP mega-kernel machinery.

The paged kernel's headline test is *bit-exactness* against the
already-verified row kernel when accumulation grouping matches — pages
are isolated as the only new variable. The mega-kernel side tests the
expert-major static bucket layout (pair conservation, placement,
determinism) and the bucket-layout oracle against a dense reference;
the remote-DMA kernel itself is tested separately (multi-device).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsv4 import eager
from dsv4.serving import attention, attention_paged as ap
from dsv4.serving import moe_megakernel as mk
from dsv4.serving import quant
from dsv4.serving.config import MoEConfig, ServingTiles

jax.config.update("jax_platform_name", "cpu")


# ---------------------------------------------------------------------------
# paged attention
# ---------------------------------------------------------------------------


@pytest.fixture()
def paged_setup():
    ks = jax.random.split(jax.random.PRNGKey(60), 8)
    B, T, n_h, c, r = 2, 4, 4, 128, 64
    S_c, S_r, n_win, P, kp = 32, 64, 16, 4, 4
    q = jax.random.normal(ks[0], (B, T, n_h, c), jnp.float32)
    kc = quant.quantize_kv(
        jax.random.normal(ks[1], (B, S_c, c), jnp.float32), r)
    swa = quant.quantize_kv(
        jax.random.normal(ks[2], (B, S_r, c), jnp.float32), r)
    sink = jax.random.normal(ks[3], (n_h,), jnp.float32) * 0.5
    pos = jnp.array([[3, 10, 20, 50], [0, 7, 33, 63]], jnp.int32)
    bound = pos // 2          # causal row bound (rows = compressed blocks)
    scores = jax.random.normal(ks[4], (B, T, S_c), jnp.float32)
    scores = jnp.where(
        jnp.arange(S_c)[None, None, :] < bound[..., None], scores, -jnp.inf)
    return q, kc, swa, sink, pos, bound, scores, n_win, P, kp, r


def test_topk_pages_causal(paged_setup):
    q, kc, swa, sink, pos, bound, scores, n_win, P, kp, r = paged_setup
    pidx = ap.topk_pages(scores, kp, P)
    # a selected page must contain >= 1 causally-valid row
    rows = ap.expand_pages_to_rows(pidx, P, bound)
    has_valid = (rows.reshape(*pidx.shape, P) >= 0).any(-1)
    assert bool(jnp.where(pidx >= 0, has_valid, True).all())
    # token with bound 0 (pos 0/1) selects nothing
    assert int(pidx[1, 0].max()) == -1


def test_paged_kernel_bit_exact_vs_row_kernel(paged_setup):
    """With matched accumulation grouping (same rows per flash update),
    the paged kernel must be BIT-EXACT vs the verified row kernel on the
    expanded indices — pages isolated as the only variable."""
    q, kc, swa, sink, pos, bound, scores, n_win, P, kp, r = paged_setup
    pidx = ap.topk_pages(scores, kp, P)
    rows = ap.expand_pages_to_rows(pidx, P, bound)
    t8 = ServingTiles(attn_chunk=8, interpret=True)   # 8 rows per update both
    out_paged = ap.sparse_mqa_paged(q, kc, pidx, bound, swa, pos, sink,
                                    page=P, n_win=n_win, rope_dim=r, tiles=t8)
    out_row = attention.sparse_mqa_gathered(q, kc, rows, swa, pos, sink,
                                            n_win=n_win, rope_dim=r, tiles=t8)
    np.testing.assert_array_equal(np.asarray(out_paged), np.asarray(out_row))


def test_paged_straddling_boundary(paged_setup):
    """A page straddling the causal bound: in-kernel masking must kill
    exactly the rows >= bound. Construct a token whose bound falls inside
    page 1 and force-select that page."""
    q, kc, swa, sink, pos, bound, scores, n_win, P, kp, r = paged_setup
    pidx = jnp.full((2, 4, kp), -1, jnp.int32).at[:, :, 0].set(1)  # page 1
    bound_mid = jnp.full((2, 4), P + 2, jnp.int32)   # rows 4..5 valid only
    t8 = ServingTiles(attn_chunk=8, interpret=True)
    out_paged = ap.sparse_mqa_paged(q, kc, pidx, bound_mid, swa, pos, sink,
                                    page=P, n_win=n_win, rope_dim=r, tiles=t8)
    rows = ap.expand_pages_to_rows(pidx, P, bound_mid)
    out_row = attention.sparse_mqa_gathered(q, kc, rows, swa, pos, sink,
                                            n_win=n_win, rope_dim=r, tiles=t8)
    np.testing.assert_array_equal(np.asarray(out_paged), np.asarray(out_row))


def test_page_recall_reported(paged_setup):
    """Recall vs row-top-k at equal row budget. Random scores are the
    structural worst case (no spatial correlation between neighbor
    blocks); real indexer scores correlate within pages → higher."""
    q, kc, swa, sink, pos, bound, scores, n_win, P, kp, r = paged_setup
    pidx = ap.topk_pages(scores, kp, P)
    row_topk = eager.topk_indices(scores, kp * P)
    rec = float(ap.page_recall(row_topk, pidx, P, bound))
    assert 0.5 < rec <= 1.0


# ---------------------------------------------------------------------------
# mega-kernel bucket layout
# ---------------------------------------------------------------------------


@pytest.fixture()
def bucket_setup():
    ks = jax.random.split(jax.random.PRNGKey(70), 6)
    M, d, E, dff, topk = 16, 256, 8, 256, 2
    cfg = MoEConfig(n_routed=E, d_expert=dff, topk=topk)
    x = jax.random.normal(ks[0], (M, d), jnp.float32) * 0.5
    w13 = jax.random.normal(ks[1], (E, d, 2 * dff), jnp.float32) * 0.05
    w2 = jax.random.normal(ks[2], (E, dff, d), jnp.float32) * 0.05
    idx = jax.random.randint(ks[3], (M, topk), 0, E).astype(jnp.int32)
    gates = jax.nn.softmax(
        jax.random.normal(ks[4], (M, topk), jnp.float32), -1)
    return cfg, x, w13, w2, idx, gates


def test_pack_dispatch_invariants(bucket_setup):
    cfg, x, w13, w2, idx, gates = bucket_setup
    M, topk = idx.shape
    b = mk.pack_dispatch(x, idx, gates, cfg, ep_size=1, n_waves=2, cap_e=8)
    # pair conservation: each routed pair exactly once (no drops at cap 8)
    ids = np.asarray(b.pair_id).reshape(-1)
    ids = ids[ids >= 0]
    assert int(b.n_drop) == 0
    assert len(ids) == M * topk and len(set(ids.tolist())) == len(ids)
    # placement: bucket (w, e) holds only pairs routed to expert w*e_wl+e
    e_wl = cfg.n_routed // 2
    pe = np.asarray(idx).reshape(-1)
    pid = np.asarray(b.pair_id)
    for w in range(2):
        for e in range(e_wl):
            for s in pid[w, 0, e].reshape(-1):
                if s >= 0:
                    assert pe[s] == w * e_wl + e
    # determinism: identical re-pack
    b2 = mk.pack_dispatch(x, idx, gates, cfg, ep_size=1, n_waves=2, cap_e=8)
    for a_, c_ in zip(jax.tree.leaves(b), jax.tree.leaves(b2)):
        np.testing.assert_array_equal(np.asarray(a_), np.asarray(c_))


def test_pack_dispatch_capacity_drop(bucket_setup):
    cfg, x, w13, w2, idx, gates = bucket_setup
    # All tokens to expert 0 -> lane overflow at small cap
    idx0 = jnp.zeros_like(idx)
    b = mk.pack_dispatch(x, idx0, gates, cfg, ep_size=1, n_waves=2, cap_e=4)
    assert int(b.n_drop) == idx.size - 4
    ids = np.asarray(b.pair_id).reshape(-1)
    assert (ids >= 0).sum() == 4


def test_bucket_reference_vs_dense_oracle(bucket_setup):
    cfg, x, w13, w2, idx, gates = bucket_setup
    M = x.shape[0]
    dff = cfg.d_expert
    b = mk.pack_dispatch(x, idx, gates, cfg, ep_size=1, n_waves=2, cap_e=8)
    out = mk.mega_moe_reference(b, w13, w2, M)
    xd = quant.dequantize_act(quant.quantize_act(x))
    ref = jnp.zeros((M, x.shape[1]), jnp.float32)
    for j in range(cfg.topk):
        for g in range(cfg.n_routed):
            sel = idx[:, j] == g
            h13 = xd @ w13[g]
            h = (h13[:, :dff] * jax.nn.sigmoid(h13[:, :dff])) * h13[:, dff:]
            hd = quant.dequantize_act(quant.quantize_act(h))
            ref += jnp.where(sel[:, None], gates[:, j:j + 1] * (hd @ w2[g]), 0.0)
    err = float(jnp.abs(out - ref).max())
    assert err / float(jnp.abs(ref).max()) < 1e-2


# ---------------------------------------------------------------------------
# mega-kernel (multi-device remote-DMA; needs forced host device count)
# ---------------------------------------------------------------------------


def _mega_run(EP, NW, E_WL, CAP, seed=80, skew=False):
    ks = jax.random.split(jax.random.PRNGKey(seed), 8)
    M_per, d, dff, topk = 8, 256, 256, 2
    E = EP * NW * E_WL
    cfg = MoEConfig(n_routed=E, d_expert=dff, topk=topk)
    M = M_per * EP
    x = jax.random.normal(ks[0], (M, d), jnp.float32) * 0.5
    w13 = jax.random.normal(ks[1], (E, d, 2 * dff), jnp.float32) * 0.05
    w2 = jax.random.normal(ks[2], (E, dff, d), jnp.float32) * 0.05
    if skew:   # everything to the experts of shard 0 (hot-shard traffic)
        idx = jax.random.randint(ks[3], (M, topk), 0, E // EP).astype(jnp.int32)
    else:
        idx = jax.random.randint(ks[3], (M, topk), 0, E).astype(jnp.int32)
    gates = jax.nn.softmax(jax.random.normal(ks[4], (M, topk), jnp.float32), -1)

    mesh = jax.make_mesh((EP,), ("ep",))
    P = jax.sharding.PartitionSpec

    def NS(s):
        return jax.sharding.NamedSharding(mesh, s)

    def shard_body(x_l, idx_l, gates_l, w13_l, w2_l):
        b = mk.pack_dispatch(x_l, idx_l, gates_l, cfg,
                             ep_size=EP, n_waves=NW, cap_e=CAP)
        return mk.mega_moe_shard(b, w13_l, w2_l, x_l.shape[0],
                                 axis_name="ep", ep_size=EP, n_waves=NW)

    fn = jax.shard_map(shard_body, mesh=mesh,
                       in_specs=(P("ep"),) * 5, out_specs=P("ep"),
                       check_vma=False)
    args = tuple(jax.device_put(a, NS(P("ep")))
                 for a in (x, idx, gates, w13, w2))
    out = jnp.asarray(jax.device_get(fn(*args)))

    xd = quant.dequantize_act(quant.quantize_act(x))
    w13b = w13.astype(jnp.bfloat16).astype(jnp.float32)
    w2b = w2.astype(jnp.bfloat16).astype(jnp.float32)
    ref = jnp.zeros((M, d), jnp.float32)
    for j in range(topk):
        for g in range(E):
            sel = idx[:, j] == g
            h13 = xd.astype(jnp.bfloat16).astype(jnp.float32) @ w13b[g]
            h = (h13[:, :dff] * jax.nn.sigmoid(h13[:, :dff])) * h13[:, dff:]
            hd = quant.dequantize_act(
                quant.quantize_act(h)).astype(jnp.bfloat16).astype(jnp.float32)
            ref += jnp.where(sel[:, None], gates[:, j:j + 1] * (hd @ w2b[g]), 0.0)
    return out, ref, fn, args


needs_devices = pytest.mark.skipif(
    jax.device_count() < 4,
    reason="needs >=4 devices "
           "(XLA_FLAGS=--xla_force_host_platform_device_count=4)")


@needs_devices
def test_megakernel_2shard_vs_oracle():
    out, ref, fn, args = _mega_run(EP=2, NW=2, E_WL=2, CAP=8)
    err = float(jnp.abs(out - ref).max())
    assert err / float(jnp.abs(ref).max()) < 3e-2
    out2 = jnp.asarray(jax.device_get(fn(*args)))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(out2))


@needs_devices
def test_megakernel_4shard_vs_oracle():
    out, ref, _, _ = _mega_run(EP=4, NW=2, E_WL=2, CAP=8, seed=81)
    err = float(jnp.abs(out - ref).max())
    assert err / float(jnp.abs(ref).max()) < 3e-2


@needs_devices
def test_megakernel_skewed_routing():
    """All pairs target shard 0's experts: exercises hot-shard recv
    pressure + empty buckets everywhere else (capacity sized to fit)."""
    out, ref, _, _ = _mega_run(EP=2, NW=2, E_WL=2, CAP=16, seed=82, skew=True)
    err = float(jnp.abs(out - ref).max())
    assert err / float(jnp.abs(ref).max()) < 3e-2
