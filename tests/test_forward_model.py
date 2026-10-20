r"""The copied forward model, its adjoint, and the backprojection. CPU only."""

import sys
from pathlib import Path

import pytest
import torch

from mrisr_vae.backprojection import backproject
from mrisr_vae.forward_model import BatchOperator, build_view_operator

from synthetic import GEOMETRY_CONFIG, geometry, make_observation, smooth_volume

VIEWS = ("sa_1", "la_2ch")
DIFFUSION_PRIORS = Path(__file__).resolve().parents[2] / "diffusion-priors"


def batch_operator(geom, views=VIEWS):
    observation = make_observation(views)
    return BatchOperator([{view: build_view_operator(observation, view, geom) for view in views}], geom.size_lvox)


def test_matches_diffusion_priors_forward_model():
    r"""Same predictions and support masks as the code this was copied from."""
    if not (DIFFUSION_PRIORS / "forward_model.py").exists():
        pytest.skip(f"{DIFFUSION_PRIORS} not found")
    sys.path.insert(0, str(DIFFUSION_PRIORS))
    try:
        import forward_model as original
    except ImportError as error:
        pytest.skip(f"cannot import diffusion-priors' forward model: {error}")
    finally:
        sys.path.remove(str(DIFFUSION_PRIORS))
    original.init_geometry(GEOMETRY_CONFIG)

    geom = geometry()
    assert geom.size_lvox == tuple(int(s) for s in original.X_SIZE_LVOXELS)
    assert torch.allclose(geom.coords_su, original.X_COORDS_SU)
    assert torch.allclose(geom.lvox_to_su_affine, original.LVOX_TO_SU_AFFINE)

    observation = make_observation(VIEWS)
    A_original = original.ForwardModel([observation], flattened=False)
    A = batch_operator(geom)
    x = smooth_volume(geom)
    with torch.no_grad():
        want = A_original(x)
        got = A(x)[0]
    assert len(got) == len(want) == len(VIEWS)
    for view, g, w in zip(VIEWS, got, want):
        assert g.shape == w.shape, view
        assert torch.allclose(g, w, atol=1e-6), f"{view}: max difference {(g - w).abs().max():.3g}"
        assert torch.equal(A.op(0, view).row_mask, A_original.fused_ops["0"][view].row_mask), view


def test_constant_volume_maps_to_constant():
    r"""A(1) = 1 on every row the latent supports, and there are such rows."""
    geom = geometry()
    A = batch_operator(geom)
    with torch.no_grad():
        predictions = A(torch.ones(1, 1, *geom.size_lvox))[0]
    for view, y_hat in zip(VIEWS, predictions):
        mask = A.op(0, view).row_mask.bool()
        assert mask.sum() > 0, f"{view}: no supported rows"
        assert torch.allclose(y_hat[0, 0][mask], torch.ones(()), atol=1e-5), view
        assert torch.all(y_hat[0, 0][~mask] == 0), view


def test_adjoint_identity_and_autograd_agree():
    r"""<A x, v> = <x, A^T v>, and A^T is what autograd uses."""
    geom = geometry()
    A = batch_operator(geom)
    generator = torch.Generator().manual_seed(1)
    x = torch.randn((1, 1, *geom.size_lvox), generator=generator).requires_grad_(True)
    predictions = A(x)[0]
    v = [torch.randn(p.shape, generator=generator) for p in predictions]

    lhs = sum(torch.sum(p * w) for p, w in zip(predictions, v))
    At_v = A.adjoint([v])
    rhs = torch.sum(x.detach() * At_v)
    assert torch.allclose(lhs.detach(), rhs, rtol=1e-4), f"{float(lhs):.6g} vs {float(rhs):.6g}"

    (grad,) = torch.autograd.grad(lhs, x)
    assert torch.allclose(grad, At_v, atol=1e-6)


def test_adjoint_of_several_channels():
    r"""The adjoint applies channel by channel, as the two-channel backprojection relies on."""
    geom = geometry()
    A = batch_operator(geom)
    shapes = [p.shape for p in A(torch.zeros(1, 1, *geom.size_lvox))[0]]
    generator = torch.Generator().manual_seed(2)
    v0 = [torch.randn(shape, generator=generator) for shape in shapes]
    v1 = [torch.randn(shape, generator=generator) for shape in shapes]
    both = A.adjoint([[torch.cat([a, b], dim=1) for a, b in zip(v0, v1)]])
    assert torch.allclose(both[:, :1], A.adjoint([v0]), atol=1e-6)
    assert torch.allclose(both[:, 1:], A.adjoint([v1]), atol=1e-6)


def test_backprojection_of_a_constant_is_that_constant():
    geom = geometry()
    A = batch_operator(geom)
    shapes = [p.shape for p in A(torch.zeros(1, 1, *geom.size_lvox))[0]]
    ys = [[torch.full(shape, 0.7) for shape in shapes]]
    variances = [[torch.tensor(0.01), torch.tensor(0.04)]]
    inp = backproject(A, ys, variances)
    assert inp.shape == (1, 2, *geom.size_lvox)
    average, coverage = inp[0, 0], inp[0, 1]
    covered = average != 0
    assert covered.sum() > 0
    assert torch.allclose(average[covered], torch.tensor(0.7), atol=1e-4)
    assert float(coverage.max()) == pytest.approx(1.0)
    assert float(coverage.min()) >= 0


def test_backprojection_is_a_precision_weighted_average():
    r"""Where views overlap, each counts in proportion to its precision."""
    geom = geometry()
    observation = make_observation(VIEWS)
    ops = {view: build_view_operator(observation, view, geom) for view in VIEWS}
    A = BatchOperator([ops], geom.size_lvox)
    shapes = [p.shape for p in A(torch.zeros(1, 1, *geom.size_lvox))[0]]
    # The long axis is 100 times noisier than the short axis.
    ys = [[torch.full(shapes[0], 1.0), torch.full(shapes[1], -1.0)]]
    average = backproject(A, ys, [[torch.tensor(0.01), torch.tensor(1.0)]])[0, 0]

    cover = {
        view: BatchOperator([{view: ops[view]}], geom.size_lvox).adjoint([[torch.ones(shape)]])[0, 0]
        for view, shape in zip(VIEWS, shapes)
    }
    sa, la = cover["sa_1"], cover["la_2ch"]
    covered = average != 0
    expected = (100 * sa - la) / (100 * sa + la)
    assert torch.allclose(average[covered], expected[covered], atol=1e-4)
    overlap = covered & (sa > 0.5 * sa.max()) & (la > 0.1 * la.max())
    assert overlap.any(), "the synthetic views do not overlap"
    assert torch.all(average[overlap] > 0.8)
