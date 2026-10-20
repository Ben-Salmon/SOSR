r"""The supervised baseline, on tiny synthetic data. CPU only.

As in test_evaluate.py, the network is shown sa_1 only and la_2ch plays the
held-out view, here its target.
"""

import copy
import csv

import pytest
import torch
import yaml

from mrisr_vae import evaluate, predict
from mrisr_vae.data import CMRObservationDataset, subjects_with_views
from mrisr_vae.model import SupervisedPredictor, build_model
from mrisr_vae.train import DEFAULTS, load_config
from mrisr_vae.trainer import SupervisedTrainer

from synthetic import GEOMETRY_CONFIG, geometry, make_item, smooth_volume, write_subject

TRAIN_CONFIG = {
    "lr": 1e-3,
    "lr_warmup_steps": 2,
    "grad_batches": 1,
    "clip": 1.0,
    "max_epochs": 2,
    "max_hours": None,
    "patience": 100,
    "plateau_patience": 20,
    "plateau_factor": 0.1,
    "monitor": "val/target_mse",
    "image_every": 1,
    "amp": False,
}


def small_model(geom):
    torch.manual_seed(0)
    return SupervisedPredictor(geom, n_layers=2, n_filters=8, downsampling=[0, 1], checkpointed=False)


def target_item(seed=0):
    r"""A synthetic item with sa_1 to show and la_2ch as the target."""
    item = make_item(geometry(), x_true=smooth_volume(geometry(), seed=seed), seed=seed, sigma=0.05)
    item["views"]["la_2ch"]["input"] = False
    return item


def test_dataset_loads_the_target_but_never_shows_it(tmp_path):
    write_subject(tmp_path)
    geom = geometry(views=["sa_1"])
    item = CMRObservationDataset(tmp_path, geom, target_views=["la_2ch"], frame=1)[0]
    assert list(item["views"]) == ["sa_1", "la_2ch"]
    assert item["views"]["sa_1"]["input"] and not item["views"]["la_2ch"]["input"]
    # The target does not set the normalisation.
    plain = CMRObservationDataset(tmp_path, geom, frame=1)[0]
    assert (item["norm_mean"], item["norm_std"]) == (plain["norm_mean"], plain["norm_std"])
    assert torch.equal(item["views"]["sa_1"]["y"], plain["views"]["sa_1"]["y"])

    # Nor is it ever an input under view dropout, which always keeps one input view.
    training = CMRObservationDataset(
        tmp_path, geom, target_views=["la_2ch"], train=True, view_drop_probability=0.99
    )
    for _ in range(20):
        item = training[0]
        assert not item["views"]["la_2ch"]["input"] and item["views"]["sa_1"]["input"]

    with pytest.raises(ValueError):
        CMRObservationDataset(tmp_path, geometry(), target_views=["la_2ch"])


def test_subjects_without_the_target_are_skipped(tmp_path):
    write_subject(tmp_path, "with")
    write_subject(tmp_path, "without", views=("sa_1",), frames=(5,))
    assert subjects_with_views(tmp_path, ["with", "without"], ["la_2ch"]) == ["with"]
    geom = geometry(views=["sa_1"])
    assert CMRObservationDataset(tmp_path, geom, subjects=["without"], target_views=["la_2ch"])[0] is None
    # Without a target, the same subject loads.
    assert CMRObservationDataset(tmp_path, geom, subjects=["without"])[0] is not None


def test_target_loss_trains_the_unet_on_the_target_alone():
    geom = geometry(views=["sa_1"])
    model = small_model(geom)
    item = target_item()
    out = model.target_loss([item], "la_2ch", amp=False)
    assert out["count"] == 1 and torch.isfinite(out["loss"]) and out["loss"] > 0
    # The encoder input is that of the shown views only.
    sa_only = {**item, "views": {"sa_1": item["views"]["sa_1"]}}
    assert torch.equal(out["input"], model.encoder_input(*model.prepare([sa_only])))
    assert out["x"].shape == (1, 1, *geom.size_lvox)

    out["loss"].backward()
    assert all(not p.requires_grad for p in model.likelihood.parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.direct.parameters())

    optimiser = torch.optim.Adamax(model.direct.parameters(), lr=1e-2)
    first = float(out["loss"])
    for _ in range(30):
        optimiser.zero_grad()
        model.target_loss([item], "la_2ch", amp=False)["loss"].backward()
        optimiser.step()
    last = float(model.target_loss([item], "la_2ch", amp=False)["loss"])
    assert last < first, f"{last:.4g} !< {first:.4g}"

    with pytest.raises(ValueError):
        model.target_loss([sa_only], "la_2ch", amp=False)


def test_supervised_trainer_fits_saves_and_resumes(tmp_path):
    geom = geometry(views=["sa_1"])
    train = [[target_item(0)], [target_item(1)]]
    val = [[target_item(2)]]
    trainer = SupervisedTrainer(small_model(geom), train, val, dict(TRAIN_CONFIG), str(tmp_path), "la_2ch")
    trainer.wandb_id = "test"
    trainer.fit()
    assert trainer.epoch == 2 and trainer.n_steps == 4
    assert trainer.direct_optimiser is None and trainer.optimisers == [trainer.optimiser]
    assert (tmp_path / "last.pt").exists() and (tmp_path / "best.pt").exists()
    assert trainer.best < float("inf")

    images = trainer.validation_images(val[0])
    # input and coverage, and the prediction
    assert len(images) == 3 * 3

    resumed = SupervisedTrainer(
        small_model(geom), train, val, dict(TRAIN_CONFIG, max_epochs=3), str(tmp_path), "la_2ch"
    )
    resumed.load(str(tmp_path / "last.pt"))
    assert resumed.epoch == 2 and resumed.n_steps == 4 and resumed.wandb_id == "test"
    for a, b in zip(resumed.model.state_dict().values(), trainer.model.state_dict().values()):
        assert torch.equal(a, b)
    resumed.fit()
    assert resumed.epoch == 3


def supervised_config(data_dir):
    config = copy.deepcopy(DEFAULTS)
    config["data"]["data_dir"] = str(data_dir)
    config["data"]["heldout_views"] = ["la_2ch"]
    config["data"]["target_view"] = "la_2ch"
    config["model"].update(type="supervised", n_layers=2, n_filters=8, downsampling=[0, 1], checkpointed=False)
    config["train"]["monitor"] = "val/target_mse"
    config["train"]["amp"] = False
    return config


def test_supervised_checkpoint_predicts_and_is_evaluated(tmp_path):
    data_dir = tmp_path / "data"
    subject = write_subject(data_dir)
    config = supervised_config(data_dir)
    torch.manual_seed(0)
    model = build_model(config, geometry(views=["sa_1"]))
    assert isinstance(model, SupervisedPredictor) and not model.can_sample
    checkpoint = tmp_path / "model.pt"
    geometry_config = dict(GEOMETRY_CONFIG, VIEWS=["sa_1"])
    torch.save({"model": model.state_dict(), "config": config, "geometry_config": geometry_config}, checkpoint)
    common = ["--checkpoint", str(checkpoint), "--subjects", subject.name, "--device", "cpu", "--num_workers", "0"]

    out_dir = tmp_path / "evaluation"
    evaluate.main([*common, "--out_dir", str(out_dir), "--save_predictions"])
    with open(out_dir / "metrics.csv") as f:
        (row,) = list(csv.DictReader(f))
    assert float(row["mse_supervised"]) > 0
    assert "mse_sample" not in row and "mse_direct" not in row
    assert (out_dir / "all" / subject.name / "la_2ch_pred_supervised.nii.gz").exists()
    with pytest.raises(SystemExit):
        evaluate.main([*common, "--out_dir", str(out_dir), "--n_samples", "2"])

    predictions = tmp_path / "predictions"
    predict.main([*common, "--out_dir", str(predictions), "--method", "direct"])
    written = sorted(path.name for path in (predictions / subject.name).iterdir())
    assert written == ["sr_supervised.nii.gz"]
    predict.main([*common, "--out_dir", str(tmp_path / "both"), "--method", "both", "--n_samples", "0"])
    assert sorted(path.name for path in (tmp_path / "both" / subject.name).iterdir()) == ["sr_supervised.nii.gz"]
    with pytest.raises(SystemExit):
        predict.main([*common, "--out_dir", str(predictions), "--method", "sampling", "--n_samples", "2"])


def write_config(path, **sections):
    config = {
        "geometry": "geometry.yaml",
        "checkpoint_dir": "checkpoints",
        "data": {"data_dir": "data", "heldout_views": ["la_4ch"], "target_view": "la_4ch"},
        "model": {"type": "supervised"},
        "train": {"monitor": "val/target_mse"},
    }
    for key, value in sections.items():
        config[key] = {**config[key], **value}
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return str(path)


def test_load_config_checks_the_supervised_options(tmp_path):
    config = load_config(write_config(tmp_path / "ok.yaml"))
    assert config["model"]["type"] == "supervised" and config["data"]["target_view"] == "la_4ch"
    bad = {
        "no_target": {"data": {"target_view": None}},
        "target_not_held_out": {"data": {"target_view": "la_2ch"}},
        "vae_monitor": {"train": {"monitor": "val/nelbo"}},
        "direct_only": {"train": {"direct_only": True}},
        "unknown_type": {"model": {"type": "diffusion"}},
        "vae_with_target": {"model": {"type": "vae"}, "train": {"monitor": "val/nelbo"}},
    }
    for name, sections in bad.items():
        with pytest.raises(ValueError):
            load_config(write_config(tmp_path / f"{name}.yaml", **sections))
