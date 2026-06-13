"""Full-model training-step tests (interpret mode — intentionally tiny).

One end-to-end gradient through the complete layer stack exercises every
custom-vjp unit in composition: sparse_mqa_train, grouped_ffn (x2 per
layer incl. shared expert), mhc_pre_norm/update, and the kernel_v2
sinkhorn — under per-layer remat.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsv4.serving import model, train_step as ts
from dsv4.serving.config import SMALL_MODEL, ServingTiles

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture(scope="module")
def setup():
    cfg = SMALL_MODEL
    tiles = ServingTiles(gemm_tm=16, attn_chunk=4, mhc_bn=8, interpret=True)
    params = ts.init_train_params(jax.random.PRNGKey(7), cfg)
    # n=24, not 16: with HCA m'=16, only tokens t >= 15 attend the single
    # compressed block, and the next-token loss drops the last position —
    # at n=16 the HCA compressor would get a structurally-zero gradient.
    toks = jax.random.randint(jax.random.PRNGKey(8), (1, 24), 0, cfg.vocab)
    return cfg, tiles, params, toks


@pytest.fixture(scope="module")
def step_result(setup):
    cfg, tiles, params, toks = setup
    return ts.train_step(params, toks, cfg, tiles=tiles, remat=True)


def test_loss_finite(step_result):
    loss, _ = step_result
    assert bool(jnp.isfinite(loss))


def test_all_grads_finite_and_trainables_nonzero(setup, step_result):
    cfg, tiles, params, toks = setup
    _, grads = step_result
    for g in jax.tree.leaves(grads):
        assert bool(jnp.isfinite(g).all())
    # SMALL_MODEL schedule: L0=swa, L1=csa, L2=hca, L3=csa.
    l0, l1, l2 = grads.layers[0], grads.layers[1], grads.layers[2]
    trainables = [
        grads.embed, grads.head,
        l1.attn.attn_sink, l1.attn.W_aKV,          # CSA sink + compressor
        l2.attn.W_KV,                              # HCA compressor
        l0.attn.W_swaK, l0.w_o1, l0.w_o2,          # swa keys + out proj
        l0.moe.w13, l0.moe.w2, l0.moe.w13_shared,  # experts (hash layer)
        l2.moe.w_router,                           # learned routing
        l0.mhc_attn.w_mix, l0.mhc_attn.scale,      # mHC via sinkhorn vjp
    ]
    for g in trainables:
        assert float(jnp.abs(g).max()) > 0.0
    # principled zero-grads under the LM loss alone (default distill_weight
    # =0): selection-only bias; indexer trains via its own distillation
    # objective (test_indexer_distillation_trains_selector), not the LM loss.
    for lg in grads.layers:
        assert float(jnp.abs(lg.moe.router_bias).max()) == 0.0
    assert float(jnp.abs(l1.attn.W_IUQ).max()) == 0.0


def test_sgd_step_decreases_loss(setup, step_result):
    cfg, tiles, params, toks = setup
    loss0, grads = step_result
    lr = 5e-2
    new_params = jax.tree.map(lambda p, g: p - lr * g.astype(p.dtype),
                              params, grads)
    loss1 = ts.lm_loss(new_params, toks, cfg, tiles=tiles, remat=True)
    assert float(loss1) < float(loss0)


def test_trained_params_serve(setup):
    """Weight-compatibility: the training tree converts to the serving
    format and runs serving prefill (the QAT story end to end)."""
    cfg, tiles, params, toks = setup
    sp = ts.to_serving_params(params, cfg)
    logits, _ = model.prefill(sp, toks, cfg,
                              model.init_state(cfg, 1, 32), tiles=tiles)
    assert logits.shape == (1, toks.shape[1], cfg.vocab)
    assert bool(jnp.isfinite(logits).all())


def test_grads_deterministic(setup, step_result):
    cfg, tiles, params, toks = setup
    loss_a, grads_a = step_result
    loss_b, grads_b = ts.train_step(params, toks, cfg, tiles=tiles,
                                    remat=True)
    np.testing.assert_array_equal(np.asarray(loss_a), np.asarray(loss_b))
    for a, b in zip(jax.tree.leaves(grads_a), jax.tree.leaves(grads_b)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_indexer_distillation_trains_selector(setup):
    """Indexer score-distillation (paper §2.3.1, train_step distill_weight>0):
    the lightning-indexer leaves that are zero-grad under the LM loss
    (W_IUQ, W_w, the ki compressor) must receive nonzero gradient, via a
    KL toward the main attention's block weights — and the distill term
    itself must be finite and non-negative (a KL). The CSA query/compressor
    paths (W_DQ shared with attention) stay trainable too."""
    cfg, tiles, params, toks = setup
    loss, grads = ts.train_step(params, toks, cfg, tiles=tiles, remat=True,
                                distill_weight=1.0)
    assert bool(jnp.isfinite(loss))
    for g in jax.tree.leaves(grads):
        assert bool(jnp.isfinite(g).all())
    l1 = grads.layers[1]                          # a CSA layer
    for name in ("W_IUQ", "W_w", "W_aIK", "W_bIK"):
        assert float(jnp.abs(getattr(l1.attn, name)).max()) > 0.0, name
    # the distillation term in isolation: finite, non-negative (KL)
    _, distill = ts.train_forward(params, toks, cfg, tiles=tiles,
                                  remat=False, distill=True)
    assert bool(jnp.isfinite(distill)) and float(distill) >= -1e-4


def test_distillation_off_leaves_indexer_zero_grad(setup):
    """distill_weight=0 is bit-identical to the pre-distillation path: the
    indexer leaves stay exactly zero-grad and the loss is unchanged."""
    cfg, tiles, params, toks = setup
    l_off, g_off = ts.train_step(params, toks, cfg, tiles=tiles, remat=True,
                                 distill_weight=0.0)
    l_base, _ = ts.train_step(params, toks, cfg, tiles=tiles, remat=True)
    np.testing.assert_array_equal(np.asarray(l_off), np.asarray(l_base))
    assert float(jnp.abs(g_off.layers[1].attn.W_IUQ).max()) == 0.0
