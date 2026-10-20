r"""The Ladder VAE, the ELBO and the training loop, on tiny synthetic data. CPU only."""

import math

import pytest
import torch

from mrisr_vae.likelihood import GaussianLikelihood
from mrisr_vae.model import MRIVAE, auto_downsampling
from mrisr_vae.networks.lvae import LadderVAE
from mrisr_vae.trainer import Trainer

from synthetic import geometry, make_item, smooth_volume

TRAIN_CONFIG = {
    "lr": 1e-3,
    "lr_warmup_steps": 2,
    "kl_warmup_steps": 0,
    "grad_batches": 1,
    "clip": 1.0,
    "free_bits": 0.0,
    "free_bits_epochs": 0,
    "max_epochs": 2,
    "max_hours": None,
    "patience": 100,
    "plateau_patience": 20,
    "plateau_factor": 0.1,
    "monitor": "val/nelbo",
    "image_every": 1,
    "amp": False,
}


def small_model(geom, checkpointed=False):
    torch.manual_seed(0)
    return MRIVAE(geom, n_layers=2, z_dim=4, n_filters=8, downsampling=[0, 1], checkpointed=checkpointed)


def test_lvae_shapes_padding_and_kl():
    torch.manual_seed(0)
    lvae = LadderVAE(
        colour_channels=2,
        img_size=(19, 19, 19),  # padded to 20 inside, cropped back
        s_code_channels=1,
        z_dims=[4, 4, 4],
        n_filters=8,
        downsampling=[[0, 0, 0], [1, 1, 1], [0, 0, 0]],
        checkpointed=True,
        dimensions=3,
    )
    x = torch.randn(2, 2, 19, 19, 19)
    out = lvae(x)
    assert out["s_code"].shape == (2, 1, 19, 19, 19)
    kl = lvae.kl_per_layer(out["q_list"], out["p_list"])
    assert kl.shape == (3,)
    assert torch.all(kl >= 0) and torch.all(torch.isfinite(kl))
    out["s_code"].sum().backward()  # through the checkpointed layers

    lvae.eval()
    with torch.no_grad():
        torch.manual_seed(1)
        a = lvae(x)["s_code"]
        torch.manual_seed(1)
        b = lvae(x)["s_code"]
        c = lvae(x)["s_code"]
    assert torch.equal(a, b)  # samples are reproducible under a fixed seed
    assert not torch.equal(a, c)
    assert lvae.sample_from_prior(3).shape == (3, 1, 19, 19, 19)


def test_auto_downsampling_follows_cosdd():
    # 83 voxels allow floor(log2(83)) - 1 = 5 halvings; with 8 layers the 3
    # that do not halve sit at even indices from the bottom.
    assert auto_downsampling((83, 83, 83), 8) == [[0] * 3, [1] * 3, [0] * 3, [1] * 3, [0] * 3, [1] * 3, [1] * 3, [1] * 3]
    # Fewer layers than halvings: every layer halves.
    assert auto_downsampling((20, 20, 20), 2) == [[1] * 3, [1] * 3]
    # Axes are treated separately; past the even indices, odd ones are used.
    assert auto_downsampling((64, 4), 4) == [[1, 0], [1, 0], [1, 0], [1, 1]]
    assert auto_downsampling((8,), 6) == [[0], [0], [0], [1], [0], [1]]


def test_model_resolves_auto_downsampling():
    geom = geometry()
    model = MRIVAE(geom, n_layers=4, z_dim=4, n_filters=8, checkpointed=False)
    assert model.downsampling == auto_downsampling(geom.size_lvox, 4)
    item = make_item(geom)
    assert torch.isfinite(model.elbo([item], amp=False)["loss"])


def test_checkpointing_does_not_change_gradients():
    r"""The recomputation draws the same latent samples as the forward pass."""
    geom = geometry()
    item = make_item(geom)
    grads = []
    for checkpointed in (False, True):
        model = small_model(geom, checkpointed=checkpointed)
        torch.manual_seed(1)
        model.elbo([item], amp=False)["loss"].backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.lvae.parameters() if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], rtol=1e-4, atol=1e-6)


def test_elbo_is_finite_and_trains_everything():
    geom = geometry()
    model = small_model(geom)
    items = [make_item(geom, seed=0), make_item(geom, views=("sa_1",), seed=1)]
    out = model.elbo(items, amp=False)
    for key in ("loss", "nelbo", "nll", "kl"):
        assert torch.isfinite(out[key]), key
    assert out["x"].shape == (2, 1, *geom.size_lvox)
    assert out["input"].shape == (2, 2, *geom.size_lvox)
    assert float(out["kl"]) >= 0
    assert torch.allclose(out["nelbo"], out["nll"] + out["kl"])
    out["loss"].backward()
    assert model.likelihood.log_scale.grad is not None
    assert torch.all(model.likelihood.log_scale.grad != 0)
    assert all(p.grad is not None for p in model.lvae.parameters() if p.requires_grad)


def test_free_bits_clamps_the_kl_term():
    geom = geometry()
    model = small_model(geom)
    item = make_item(geom)
    torch.manual_seed(0)
    plain = model.elbo([item], amp=False)
    torch.manual_seed(0)
    clamped = model.elbo([item], amp=False, free_bits=1e3)
    assert torch.allclose(clamped["loss"], plain["nll"] + 1e3)
    assert torch.allclose(clamped["nelbo"], plain["nelbo"])


def test_kl_weight_scales_the_kl_term():
    geom = geometry()
    model = small_model(geom)
    item = make_item(geom)
    torch.manual_seed(0)
    plain = model.elbo([item], amp=False)
    torch.manual_seed(0)
    weighted = model.elbo([item], amp=False, kl_weight=0.25)
    assert torch.allclose(weighted["loss"], plain["nll"] + 0.25 * plain["kl"])
    assert torch.allclose(weighted["nelbo"], plain["nelbo"])


def test_views_dropped_from_the_input_stay_in_the_likelihood():
    geom = geometry()
    model = small_model(geom)
    item = make_item(geom, seed=0)
    dropped = {**item, "views": {**item["views"], "la_2ch": {**item["views"]["la_2ch"], "input": False}}}
    sa_only = {**item, "views": {"sa_1": item["views"]["sa_1"]}}
    out = model.elbo([dropped], amp=False)
    assert torch.equal(out["input"], model.encoder_input(*model.prepare([sa_only])))
    assert not torch.equal(out["input"], model.elbo([item], amp=False)["input"])
    assert out["n_rows"] == model.elbo([item], amp=False)["n_rows"]
    assert out["n_rows"] > model.elbo([sa_only], amp=False)["n_rows"]


def test_kl_warmup_ramps_the_weight_and_holds_back_the_schedules(tmp_path):
    geom = geometry()
    trainer = Trainer(
        small_model(geom), [], [], dict(TRAIN_CONFIG, kl_warmup_steps=4), noise_lr=0.01, checkpoint_dir=str(tmp_path)
    )
    weights = []
    for n_steps in (0, 2, 4, 10):
        trainer.n_steps = n_steps
        weights.append((trainer.kl_weight(), trainer.in_warmup()))
    assert weights == [(0.0, True), (0.5, True), (1.0, False), (1.0, False)]
    trainer.cfg["kl_warmup_steps"] = 0
    trainer.n_steps = 0
    assert trainer.kl_weight() == 1.0


def test_likelihood_noise_levels():
    likelihood = GaussianLikelihood(["sa_1", "sa_2", "la_2ch"], learn=True, min_sigma=0.0)
    assert likelihood.groups == ["la", "sa"]
    sigma = likelihood.sigma("sa_2", torch.tensor(0.3), torch.tensor(0.4))
    assert float(sigma) == pytest.approx(0.5)
    with torch.no_grad():
        likelihood.log_scale[likelihood.groups.index("la")] = math.log(2.0)
    assert float(likelihood.sigma("la_2ch", torch.tensor(0.0), torch.tensor(0.4))) == pytest.approx(0.8)
    assert float(likelihood.sigma("sa_1", torch.tensor(0.0), torch.tensor(0.4))) == pytest.approx(0.4)

    y = torch.zeros(1, 1, 2, 2, 1)
    y_hat = torch.ones(1, 1, 2, 2, 1)
    mask = torch.tensor([[[1.0], [1.0]], [[1.0], [0.0]]])
    nll, n = likelihood.nll(y, y_hat, mask, torch.tensor(1.0))
    assert float(n) == 3
    assert float(nll) == pytest.approx(3 * (0.5 + 0.5 * math.log(2 * math.pi)))


def test_global_noise_levels():
    likelihood = GaussianLikelihood(
        ["sa_1", "la_2ch"], noise="global", init_sigma=0.2, relative_std={"la_2ch": 4.0}, min_sigma=0.0
    )
    # Every view shares sigma; la gets 4x it, and sigma_extra is not used.
    assert float(likelihood.sigma("sa_1", torch.tensor(0.0), torch.tensor(9.0))) == pytest.approx(0.2)
    assert float(likelihood.sigma("la_2ch", torch.tensor(0.0), torch.tensor(9.0))) == pytest.approx(0.8)
    assert float(likelihood.sigma("sa_1", torch.tensor(0.15), torch.tensor(9.0))) == pytest.approx(0.25)
    # The encoder input weights views by the fixed ratio, not the learned sigma.
    with torch.no_grad():
        likelihood.log_sigma.fill_(math.log(0.01))
    assert float(likelihood.input_variance("la_2ch", torch.tensor(0.0), torch.tensor(9.0))) == pytest.approx(0.8**2)
    assert likelihood.log_values() == {"noise_sigma": pytest.approx(0.01)}
    with pytest.raises(ValueError):
        GaussianLikelihood(["sa_1"], noise="per_view")


def test_global_noise_is_learned_from_every_view():
    geom = geometry()
    torch.manual_seed(0)
    model = MRIVAE(
        geom, n_layers=2, z_dim=4, n_filters=8, downsampling=[0, 1], checkpointed=False,
        noise="global", init_sigma=0.1, relative_std={"la_2ch": 4.0},
    )
    assert not hasattr(model.likelihood, "log_scale")
    out = model.elbo([make_item(geom, seed=0)], amp=False)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.likelihood.log_sigma.grad is not None and model.likelihood.log_sigma.grad != 0


def test_training_reduces_the_negative_elbo():
    geom = geometry()
    model = small_model(geom)
    item = make_item(geom, x_true=smooth_volume(geom), sigma=0.05)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    torch.manual_seed(0)
    first = float(model.elbo([item], amp=False)["nelbo"])
    for _ in range(60):
        loss = model.elbo([item], amp=False)["loss"]
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        last = float(model.elbo([item], amp=False)["nelbo"])
    assert last < first, f"{last:.4g} !< {first:.4g}"


def test_heldout_error_holds_out_the_last_view():
    geom = geometry()
    model = small_model(geom).eval()
    items = [make_item(geom, seed=0), make_item(geom, views=("sa_1",), seed=1)]
    torch.manual_seed(0)
    total, count = model.heldout_error(items, amp=False)
    assert count == 1  # the single-view observation has nothing to hold out
    assert torch.isfinite(total) and float(total) > 0
    torch.manual_seed(0)
    assert torch.equal(model.heldout_error(items, amp=False)[0], total)  # reproducible under a fixed seed


def test_posterior_and_prior_samples():
    geom = geometry()
    model = small_model(geom).eval()
    item = make_item(geom)
    samples = model.posterior([item], n_samples=3, amp=False)
    assert samples.shape == (3, 1, 1, *geom.size_lvox)
    assert not torch.equal(samples[0], samples[1])
    assert model.sample_prior(2, amp=False).shape == (2, 1, *geom.size_lvox)


def test_direct_predictor_trains_on_detached_samples_only():
    geom = geometry()
    model = small_model(geom)
    item = make_item(geom, x_true=smooth_volume(geom), sigma=0.05)
    out = model.elbo([item], amp=False)
    direct_mse = model.direct_loss(out["input"], out["x"], amp=False)
    assert torch.isfinite(direct_mse)
    # Its loss reaches the predictor and nothing else, even given an undetached target.
    direct_mse.backward()
    assert all(p.grad is None for p in model.lvae.parameters())
    assert model.likelihood.log_scale.grad is None
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.direct.parameters())

    prediction = model.predict_direct([item], amp=False)
    assert prediction.shape == (1, 1, *geom.size_lvox)
    total, count = model.heldout_error([item], amp=False, direct=True)
    assert count == 1 and torch.isfinite(total)

    # Regression onto samples of a fixed VAE brings the prediction to their mean.
    optimiser = torch.optim.Adam(model.direct.parameters(), lr=3e-3)
    model.eval()
    with torch.no_grad():
        target = model.posterior([item], n_samples=16, amp=False).mean(0)
    first = float(torch.mean((model.predict_direct([item], amp=False) - target) ** 2))
    for _ in range(60):
        with torch.no_grad():
            out = model.elbo([item], amp=False)
        loss = model.direct_loss(out["input"], out["x"], amp=False)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    last = float(torch.mean((model.predict_direct([item], amp=False) - target) ** 2))
    assert last < first, f"{last:.4g} !< {first:.4g}"


def test_model_without_direct_predictor(tmp_path):
    geom = geometry()
    model = MRIVAE(geom, n_layers=2, z_dim=4, n_filters=8, downsampling=[0, 1], direct_predictor=False)
    assert model.direct is None
    out = model.elbo([make_item(geom)], amp=False)
    with pytest.raises(RuntimeError):
        model.direct_loss(out["input"], out["x"], amp=False)
    trainer = Trainer(model, [], [], dict(TRAIN_CONFIG), noise_lr=0.01, checkpoint_dir=str(tmp_path))
    assert trainer.direct_optimiser is None and len(trainer.optimisers) == 1


def test_trainer_fits_saves_and_resumes(tmp_path):
    geom = geometry()
    train = [[make_item(geom, seed=0)], [make_item(geom, views=("sa_1",), seed=1)]]
    val = [[make_item(geom, seed=2)]]

    trainer = Trainer(small_model(geom), train, val, dict(TRAIN_CONFIG), noise_lr=0.01, checkpoint_dir=str(tmp_path))
    trainer.wandb_id = "test"
    trainer.fit()
    assert trainer.epoch == 2
    assert trainer.n_steps == 4
    # Both optimisers stepped once per step.
    for optimiser in trainer.optimisers:
        steps = [state["step"] for state in optimiser.state.values() if "step" in state]
        assert steps and all(int(s) == 4 for s in steps)
    assert (tmp_path / "last.pt").exists() and (tmp_path / "best.pt").exists()

    images = trainer.validation_images(val[0])
    # input and coverage, two posterior samples, a prior sample, the direct prediction
    assert len(images) == 6 * 3
    assert all(image.dtype.name == "uint8" and image.ndim == 2 for image in images.values())

    resumed = Trainer(
        small_model(geom), train, val, dict(TRAIN_CONFIG, max_epochs=3), noise_lr=0.01, checkpoint_dir=str(tmp_path)
    )
    resumed.load(str(tmp_path / "last.pt"))
    assert resumed.epoch == 2 and resumed.n_steps == 4 and resumed.wandb_id == "test"
    for a, b in zip(resumed.model.state_dict().values(), trainer.model.state_dict().values()):
        assert torch.equal(a, b)
    resumed.fit()
    assert resumed.epoch == 3
