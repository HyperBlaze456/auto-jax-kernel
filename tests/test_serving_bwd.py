"""Backward-pass correctness suite for dsv4.serving (interpret mode, CPU).

Every custom gradient is graded against ``jax.grad`` of an STE-modeled
dense fp32 oracle — the oracle encodes the *same* quantization points
(fp8 act/weight/hidden with straight-through estimators), so agreement is
bounded only by the declared bf16-gradient policy (≲1% rel), never by
kernel slop. Determinism tests assert bitwise reproducibility — the TPU
analogue of paper §3.3's deterministic-accumulation requirement.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsv4.serving import attention_train as at
from dsv4.serving import gemm_fp8_diff as gd
from dsv4.serving import mhc, mhc_diff, moe_diff
from dsv4.serving.config import MoEConfig, ServingTiles

jax.config.update("jax_platform_name", "cpu")

BF16_POLICY_RTOL = 2e-2   # bf16 grads + fp8 operand effects, vs fp32 oracle


def _rel(a, b):
    sc = float(jnp.abs(b).max()) + 1e-12
    return float(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)).max()) / sc


# ---------------------------------------------------------------------------
# grouped FFN (Linear1 + SwiGLU + fp8 cast + Linear2)
# ---------------------------------------------------------------------------


@pytest.fixture()
def ffn_setup():
    ks = jax.random.split(jax.random.PRNGKey(20), 5)
    M, K, dff, N, E = 128, 256, 256, 256, 4
    x = jax.random.normal(ks[0], (M, K), jnp.float32) * 0.5
    w13 = jax.random.normal(ks[1], (E, K, 2 * dff), jnp.float32) * 0.05
    w2 = jax.random.normal(ks[2], (E, dff, N), jnp.float32) * 0.05
    groups = jnp.array([30, 0, 60, 26], jnp.int32)   # empty group + padding
    cot = jax.random.normal(ks[3], (M, N), jnp.float32).at[
        int(groups.sum()):].set(0.0)
    return x, w13, w2, groups, cot


def test_grouped_ffn_forward(ffn_setup):
    x, w13, w2, groups, _ = ffn_setup
    v = int(groups.sum())
    y = gd.grouped_ffn(x, w13, w2, groups, tm=16, interpret=True)
    y_ref = gd.grouped_ffn_ref(x, w13, w2, groups)
    assert _rel(y[:v], y_ref[:v]) < 1e-2


def test_grouped_ffn_grads(ffn_setup):
    x, w13, w2, groups, cot = ffn_setup
    v = int(groups.sum())

    def loss_k(x, w13, w2):
        y = gd.grouped_ffn(x, w13, w2, groups, tm=16, interpret=True)
        return (y.astype(jnp.float32) * cot).sum()

    def loss_r(x, w13, w2):
        return (gd.grouped_ffn_ref(x, w13, w2, groups) * cot).sum()

    gk = jax.grad(loss_k, argnums=(0, 1, 2))(x, w13, w2)
    gr = jax.grad(loss_r, argnums=(0, 1, 2))(x, w13, w2)
    assert _rel(gk[0][:v], gr[0][:v]) < BF16_POLICY_RTOL     # dx
    assert _rel(gk[1], gr[1]) < BF16_POLICY_RTOL             # dw13
    assert _rel(gk[2], gr[2]) < BF16_POLICY_RTOL             # dw2


def test_grouped_ffn_bwd_deterministic(ffn_setup):
    x, w13, w2, groups, cot = ffn_setup

    def loss(x, w13, w2):
        y = gd.grouped_ffn(x, w13, w2, groups, tm=16, interpret=True)
        return (y.astype(jnp.float32) * cot).sum()

    g1 = jax.grad(loss, argnums=(0, 1, 2))(x, w13, w2)
    g2 = jax.grad(loss, argnums=(0, 1, 2))(x, w13, w2)
    for a, b in zip(g1, g2):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# gather attention (training path)
# ---------------------------------------------------------------------------


@pytest.fixture()
def attn_setup():
    ks = jax.random.split(jax.random.PRNGKey(30), 8)
    B, T, n_h, c = 2, 4, 4, 128
    S_c, S_r, k = 32, 64, 8
    q = jax.random.normal(ks[0], (B, T, n_h, c), jnp.float32) * 0.5
    kc = jax.random.normal(ks[1], (B, S_c, c), jnp.float32) * 0.5
    sw = jax.random.normal(ks[2], (B, S_r, c), jnp.float32) * 0.5
    sink = jax.random.normal(ks[3], (n_h,), jnp.float32) * 0.3
    idx = jax.random.randint(ks[4], (B, T, k), 0, S_c)
    # -1 padding + heavy duplicate indices across tokens (scatter coverage)
    idx = jnp.where(jax.random.bernoulli(ks[5], 0.25, (B, T, k)), -1,
                    idx % 16).astype(jnp.int32)
    pos = jnp.array([[3, 10, 20, 50], [0, 7, 33, 63]], jnp.int32)
    cot = jax.random.normal(ks[6], (B, T, n_h, c), jnp.float32)
    tiles = ServingTiles(attn_chunk=4, interpret=True)
    return q, kc, sw, sink, idx, pos, cot, tiles


def test_attn_train_forward(attn_setup):
    q, kc, sw, sink, idx, pos, cot, tiles = attn_setup
    o = at.sparse_mqa_train(q, kc, idx, sw, pos, sink, n_win=16, tiles=tiles)
    o_ref = at.sparse_mqa_train_ref(q, kc, idx, sw, pos, sink, n_win=16)
    assert float(jnp.abs(o.astype(jnp.float32) - o_ref).max()) < 1e-2


def test_attn_train_grads(attn_setup):
    q, kc, sw, sink, idx, pos, cot, tiles = attn_setup

    def loss_k(q, kc, sw, sink):
        o = at.sparse_mqa_train(q, kc, idx, sw, pos, sink, n_win=16,
                                tiles=tiles)
        return (o.astype(jnp.float32) * cot).sum()

    def loss_r(q, kc, sw, sink):
        o = at.sparse_mqa_train_ref(q, kc, idx, sw, pos, sink, n_win=16)
        return (o * cot).sum()

    gk = jax.grad(loss_k, argnums=(0, 1, 2, 3))(q, kc, sw, sink)
    gr = jax.grad(loss_r, argnums=(0, 1, 2, 3))(q, kc, sw, sink)
    for a, b in zip(gk, gr):                 # dq, dK_comp, dK_swa, dsink
        assert _rel(a, b) < BF16_POLICY_RTOL


def test_attn_train_bwd_deterministic(attn_setup):
    q, kc, sw, sink, idx, pos, cot, tiles = attn_setup

    def loss(q, kc, sw, sink):
        o = at.sparse_mqa_train(q, kc, idx, sw, pos, sink, n_win=16,
                                tiles=tiles)
        return (o.astype(jnp.float32) * cot).sum()

    g1 = jax.grad(loss, argnums=(0, 1, 2, 3))(q, kc, sw, sink)
    g2 = jax.grad(loss, argnums=(0, 1, 2, 3))(q, kc, sw, sink)
    for a, b in zip(g1, g2):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# mHC fused backward
# ---------------------------------------------------------------------------


def test_mhc_diff_grads():
    ks = jax.random.split(jax.random.PRNGKey(40), 6)
    B, n, hc, d = 2, 12, 4, 256
    tiles = ServingTiles(mhc_bn=8, interpret=True)   # n % bn != 0 → padding
    x = jax.random.normal(ks[0], (B, n, hc, d), jnp.float32)
    pre = jax.nn.sigmoid(jax.random.normal(ks[1], (B, n, hc), jnp.float32))
    comb = jax.nn.softmax(
        jax.random.normal(ks[2], (B, n, hc, hc), jnp.float32), -1)
    post = 2 * jax.nn.sigmoid(jax.random.normal(ks[3], (B, n, hc), jnp.float32))
    f = jax.random.normal(ks[4], (B, n, d), jnp.float32)
    cot_h = jax.random.normal(ks[5], (B, n, d), jnp.float32)
    cot_x = jax.random.normal(ks[5], (B, n, hc, d), jnp.float32)

    gk = jax.grad(lambda x, p: (
        mhc_diff.mhc_pre_norm_diff(x, p, tiles=tiles) * cot_h).sum(),
        argnums=(0, 1))(x, pre)
    gr = jax.grad(lambda x, p: (
        mhc.mhc_pre_norm_ref(x, p) * cot_h).sum(), argnums=(0, 1))(x, pre)
    for a, b in zip(gk, gr):
        assert float(jnp.abs(a - b).max()) < 1e-4

    gk = jax.grad(lambda *a: (
        mhc_diff.mhc_update_diff(*a, tiles=tiles) * cot_x).sum(),
        argnums=(0, 1, 2, 3))(x, comb, post, f)
    gr = jax.grad(lambda *a: (
        mhc.mhc_update_ref(*a) * cot_x).sum(),
        argnums=(0, 1, 2, 3))(x, comb, post, f)
    for a, b in zip(gk, gr):
        assert float(jnp.abs(a - b).max()) < 1e-4


# ---------------------------------------------------------------------------
# full diffable MoE layer
# ---------------------------------------------------------------------------


def test_moe_diff_full_gradient_tree():
    ks = jax.random.split(jax.random.PRNGKey(50), 8)
    M, d, E, dff, topk = 24, 256, 4, 256, 2
    cfg = MoEConfig(n_routed=E, d_expert=dff, topk=topk)
    tiles = ServingTiles(gemm_tm=16, interpret=True)
    params = moe_diff.MoEParamsTrain(
        w_router=jax.random.normal(ks[0], (d, E), jnp.float32) * 0.1,
        router_bias=jnp.zeros((E,), jnp.float32),
        w13=jax.random.normal(ks[1], (E, d, 2 * dff), jnp.float32) * 0.05,
        w2=jax.random.normal(ks[2], (E, dff, d), jnp.float32) * 0.05,
        w13_shared=jax.random.normal(ks[3], (1, d, 2 * dff), jnp.float32) * 0.05,
        w2_shared=jax.random.normal(ks[4], (1, dff, d), jnp.float32) * 0.05,
    )
    x = jax.random.normal(ks[5], (M, d), jnp.float32) * 0.5
    cot = jax.random.normal(ks[6], (M, d), jnp.float32)

    def loss_k(x, p):
        y = moe_diff.moe_forward_local_diff(x, p, cfg, tiles=tiles)
        return (y.astype(jnp.float32) * cot).sum()

    def loss_r(x, p):
        y = moe_diff.moe_forward_local_diff_ref(x, p, cfg)
        return (y.astype(jnp.float32) * cot).sum()

    gk = jax.grad(loss_k, argnums=(0, 1))(x, params)
    gr = jax.grad(loss_r, argnums=(0, 1))(x, params)
    assert _rel(gk[0], gr[0]) < BF16_POLICY_RTOL
    for name, a, b in zip(params._fields, gk[1], gr[1]):
        if name == "router_bias":
            # selection-only (aux-loss-free): must carry no gradient
            assert float(jnp.abs(a).max()) == 0.0
            continue
        assert _rel(a, b) < BF16_POLICY_RTOL, name


def test_moe_ep_diff_matches_local_diff():
    """Training-path EP (HARDWARE_NOTES §13.11): moe_forward_ep_diff under
    shard_map must match the local diff path in both forward and the input
    gradient — i.e. the dispatch/combine all_to_all's autodiff transparently
    (dispatch-bwd is a combine-shaped a2a and vice versa). Needs >=2 devices
    (XLA_FLAGS=--xla_force_host_platform_device_count=4)."""
    from functools import partial
    if jax.device_count() < 2:
        pytest.skip("needs >=2 devices")
    ks = jax.random.split(jax.random.PRNGKey(77), 8)
    M, d, E, dff, topk = 16, 256, 4, 256, 2
    EP, NW = 2, 2
    cfg = MoEConfig(n_routed=E, d_expert=dff, topk=topk)
    tiles = ServingTiles(gemm_tm=16, interpret=True)
    params = moe_diff.MoEParamsTrain(
        w_router=jax.random.normal(ks[0], (d, E), jnp.float32) * 0.1,
        router_bias=jnp.zeros((E,), jnp.float32),
        w13=jax.random.normal(ks[1], (E, d, 2 * dff), jnp.float32) * 0.05,
        w2=jax.random.normal(ks[2], (E, dff, d), jnp.float32) * 0.05,
        w13_shared=jax.random.normal(ks[3], (1, d, 2 * dff), jnp.float32) * 0.05,
        w2_shared=jax.random.normal(ks[4], (1, dff, d), jnp.float32) * 0.05,
    )
    x = jax.random.normal(ks[5], (M, d), jnp.float32) * 0.5
    cot = jax.random.normal(ks[6], (M, d), jnp.float32)
    idx, gates = moe_diff.route_train(x, params, cfg)   # shared selection

    def loss_local(x):
        y = moe_diff.moe_forward_local_diff(x, params, cfg, tiles=tiles,
                                            idx_gates=(idx, gates))
        return (y.astype(jnp.float32) * cot).sum()
    out_local = moe_diff.moe_forward_local_diff(
        x, params, cfg, tiles=tiles, idx_gates=(idx, gates))
    dx_local = jax.grad(loss_local)(x)

    mesh = jax.make_mesh((EP,), ("ep",))
    P = jax.sharding.PartitionSpec
    NS = lambda s: jax.sharding.NamedSharding(mesh, s)
    ep_specs = moe_diff.MoEParamsTrain(
        w_router=P(), router_bias=P(), w13=P("ep"), w2=P("ep"),
        w13_shared=P(), w2_shared=P())
    cap = (M // EP) * topk
    ep_params = jax.tree.map(lambda a, s: jax.device_put(a, NS(s)),
                             params, ep_specs)
    xs = jax.device_put(x, NS(P("ep")))
    idxs = jax.device_put(idx, NS(P("ep")))
    gs = jax.device_put(gates, NS(P("ep")))
    cots = jax.device_put(cot, NS(P("ep")))

    def ep_apply(x_, idx_, g_, p_):
        fn = jax.shard_map(
            partial(moe_diff.moe_forward_ep_diff, cfg=cfg, axis_name="ep",
                    ep_size=EP, n_waves=NW, capacity=cap, tiles=tiles),
            mesh=mesh, in_specs=(P("ep"), P("ep"), P("ep"), ep_specs),
            out_specs=P("ep"), check_vma=False)
        return fn(x_, idx_, g_, p_)

    def loss_ep(x_):
        return (ep_apply(x_, idxs, gs, ep_params).astype(jnp.float32)
                * cots).sum()

    with jax.set_mesh(mesh):                       # sharded-scalar reduce/grad
        out_ep = ep_apply(xs, idxs, gs, ep_params)
        dx_ep = jax.grad(loss_ep)(xs)
    np.testing.assert_allclose(np.asarray(out_ep), np.asarray(out_local),
                               rtol=BF16_POLICY_RTOL, atol=2e-2)
    assert _rel(dx_ep, dx_local) < BF16_POLICY_RTOL


def test_segment_reduce_sorted_matches_reference():
    """The §13.12 sort-by-destination reduction: per destination row, the
    sum of its contributions — equal to a plain scatter-add (numpy
    groupby) up to f32 reassociation, deterministic, and -1 entries
    dropped."""
    ks = jax.random.split(jax.random.PRNGKey(91), 2)
    B, N, c, n_dest = 2, 40, 16, 8
    dest = jax.random.randint(ks[0], (B, N), -1, n_dest).astype(jnp.int32)
    contrib = jax.random.normal(ks[1], (B, N, c), jnp.float32)
    got = at._segment_reduce_sorted(dest, contrib, n_dest)
    assert got.shape == (B, n_dest, c)
    ref = np.zeros((B, n_dest, c), np.float64)
    d, ct = np.asarray(dest), np.asarray(contrib, np.float64)
    for b in range(B):
        for i in range(N):
            if d[b, i] >= 0:
                ref[b, d[b, i]] += ct[b, i]
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-4, atol=1e-4)
    # deterministic
    np.testing.assert_array_equal(
        np.asarray(got), np.asarray(at._segment_reduce_sorted(dest, contrib, n_dest)))
