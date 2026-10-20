r"""Held-out-view evaluation, end to end on a synthetic subject. CPU only.

Here the model is trained on sa_1 only and la_2ch plays the held-out view.
"""

import copy
import csv

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from mrisr_vae import evaluate
from mrisr_vae.evaluate import resolve_conditioning
from mrisr_vae.metrics import score_directory
from mrisr_vae.model import build_model
from mrisr_vae.train import DEFAULTS, check_heldout_views, check_relative_std

from synthetic import GEOMETRY_CONFIG, geometry, write_subject


def small_config(data_dir, heldout_views=("la_2ch",)):
    config = copy.deepcopy(DEFAULTS)
    config["data"]["data_dir"] = str(data_dir)
    config["data"]["heldout_views"] = list(heldout_views)
    config["data"]["extra_noise_std"] = {"sa_1": 4.0, "la_2ch": 20.0}
    config["model"].update(n_layers=2, z_dim=4, n_filters=8, downsampling=[0, 1], checkpointed=False)
    config["train"]["amp"] = False
    return config


def test_training_refuses_a_heldout_view_in_views(tmp_path):
    check_heldout_views(small_config(tmp_path), geometry(views=["sa_1"]))
    with pytest.raises(ValueError):
        check_heldout_views(small_config(tmp_path), geometry(views=["sa_1", "la_2ch"]))


def test_relative_std_must_name_known_views(tmp_path):
    config = small_config(tmp_path)
    # la_2ch is only held out here, which is allowed.
    config["likelihood"]["relative_std"] = {"la_2ch": 4.0}
    check_relative_std(config, geometry(views=["sa_1"]))
    config["likelihood"]["relative_std"] = {"la": 4.0}
    with pytest.raises(ValueError):
        check_relative_std(config, geometry(views=["sa_1"]))


def test_evaluate_scores_and_saves_the_heldout_view(tmp_path):
    data_dir = tmp_path / "data"
    subject = write_subject(data_dir)
    geometry_config = dict(GEOMETRY_CONFIG, VIEWS=["sa_1"])
    config = small_config(data_dir)
    torch.manual_seed(0)
    model = build_model(config, geometry(views=["sa_1"]))
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "config": config, "geometry_config": geometry_config}, checkpoint)

    out_dir = tmp_path / "evaluation"
    evaluate.main(
        [
            "--checkpoint", str(checkpoint),
            "--out_dir", str(out_dir),
            "--subjects", subject.name,
            "--conditioning_views", "all", "sa",
            "--n_samples", "2",
            "--save_predictions",
            "--device", "cpu",
            "--num_workers", "0",
        ]
    )

    with open(out_dir / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    assert [row["conditioning"] for row in rows] == ["all", "sa"]
    assert rows[0]["frame"] == rows[1]["frame"]  # every conditioning set on the same frame
    row = rows[0]
    assert row["view"] == "la_2ch" and row["conditioning_views"] == "sa_1"
    for key in ("mse_sample", "mse_sample_mean", "mse_direct"):
        assert np.isfinite(float(row[key])) and float(row[key]) > 0, key
    for key in ("psnr_sample", "psnr_sample_mean", "psnr_direct"):
        assert np.isfinite(float(row[key])), key
    # The mean of the samples cannot be further from the target than the samples on average.
    assert float(row["mse_sample_mean"]) <= float(row["mse_sample"]) * (1 + 1e-6)
    # The projected samples' spread, and its rank correlation with the error.
    assert float(row["mean_std"]) > 0
    assert -1 <= float(row["spearman_std_error"]) <= 1

    raw = nib.load(str(subject / "la_2ch.nii.gz"))
    frame = int(row["frame"])
    subject_dir = out_dir / "all" / subject.name
    for name in ("target", "pred_sample_mean", "pred_direct", "mask"):
        nifti = nib.load(str(subject_dir / f"la_2ch_{name}.nii.gz"))
        assert nifti.shape == raw.shape[:3], name
        assert np.allclose(nifti.affine, raw.affine), name
        assert int(nifti.header["toffset"]) == frame, name
    mask = nib.load(str(subject_dir / "la_2ch_mask.nii.gz")).get_fdata() > 0
    assert mask.sum() == int(row["n_voxels"])
    std = nib.load(str(subject_dir / "la_2ch_sample_std.nii.gz"))
    assert std.shape == raw.shape[:3] and np.allclose(std.affine, raw.affine)
    std = std.get_fdata()
    assert np.all(std[~mask] == 0) and np.all(std[mask] >= 0)
    assert np.mean(std[mask]) == pytest.approx(float(row["mean_std"]), rel=1e-4)
    # It is not a prediction, so mrisr-score finds nothing to score under its name.
    assert score_directory(out_dir, "sample_std") == []
    target = nib.load(str(subject_dir / "la_2ch_target.nii.gz")).get_fdata()
    assert np.allclose(target[mask], raw.get_fdata()[..., frame][mask], atol=1e-2)

    # Another method saved beside them is scored by the same code: a copy of
    # the direct prediction scores as the direct prediction did.
    direct = nib.load(str(subject_dir / "la_2ch_pred_direct.nii.gz"))
    nib.save(direct, str(subject_dir / "la_2ch_pred_baseline.nii.gz"))
    scores = [s for s in score_directory(out_dir, "baseline") if s["conditioning"] == "all"]
    assert len(scores) == 1
    assert scores[0]["mse"] == pytest.approx(float(row["mse_direct"]), rel=1e-3)
    assert scores[0]["psnr"] == pytest.approx(float(row["psnr_direct"]), rel=1e-3)
    if np.isfinite(float(row["ssim_direct"])):
        assert scores[0]["ssim"] == pytest.approx(float(row["ssim_direct"]), abs=1e-3)


def test_conditioning_sets():
    views = ["sa_1", "sa_2", "la_2ch"]
    assert resolve_conditioning("all", views) == views
    assert resolve_conditioning("sa", views) == ["sa_1", "sa_2"]
    assert resolve_conditioning("la_2ch", views) == ["la_2ch"]
    assert resolve_conditioning("la_2ch,sa_2", views) == ["sa_2", "la_2ch"]
    with pytest.raises(ValueError):
        resolve_conditioning("la_4ch", views)


def test_evaluate_refuses_a_trained_view(tmp_path):
    data_dir = tmp_path / "data"
    write_subject(data_dir)
    config = small_config(data_dir, heldout_views=("sa_1",))
    model = build_model(config, geometry(views=["sa_1"]))
    checkpoint = tmp_path / "model.pt"
    geometry_config = dict(GEOMETRY_CONFIG, VIEWS=["sa_1"])
    torch.save({"model": model.state_dict(), "config": config, "geometry_config": geometry_config}, checkpoint)
    with pytest.raises(SystemExit):
        evaluate.main(["--checkpoint", str(checkpoint), "--out_dir", str(tmp_path / "out"), "--device", "cpu"])


def save_checkpoint(path, config):
    torch.manual_seed(0)
    model = build_model(config, geometry(views=["sa_1"]))
    geometry_config = dict(GEOMETRY_CONFIG, VIEWS=["sa_1"])
    torch.save({"model": model.state_dict(), "config": config, "geometry_config": geometry_config}, path)


def read_rows(out_dir):
    with open(out_dir / "metrics.csv") as f:
        return list(csv.DictReader(f))


def test_samples_depend_only_on_the_seed_subject_and_conditioning(tmp_path):
    data_dir = tmp_path / "data"
    # subject_a sorts first, so it runs before subject_z when both are evaluated.
    write_subject(data_dir, "subject_z", seed=0)
    write_subject(data_dir, "subject_a", seed=1)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, small_config(data_dir))

    def run(name, *options):
        out_dir = tmp_path / name
        evaluate.main(
            ["--checkpoint", str(checkpoint), "--out_dir", str(out_dir), "--n_samples", "2",
             "--device", "cpu", "--num_workers", "0", *options]
        )
        return {row["subject"]: row for row in read_rows(out_dir)}

    alone = run("alone", "--subjects", "subject_z")
    together = run("together", "--subjects", "subject_z", "subject_a")
    # The same samples for subject_z whether or not subject_a was sampled before it.
    for key in ("mse_sample", "mse_sample_mean", "mean_std", "spearman_std_error"):
        assert alone["subject_z"][key] == together["subject_z"][key], key
    other_seed = run("other_seed", "--subjects", "subject_z", "--seed", "1")
    assert other_seed["subject_z"]["mse_sample"] != alone["subject_z"]["mse_sample"]
    # The direct predictor is deterministic.
    assert other_seed["subject_z"]["mse_direct"] == alone["subject_z"]["mse_direct"]


def test_evaluate_reads_its_config(tmp_path):
    data_dir = tmp_path / "test_data"
    subject = write_subject(data_dir)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    # Trained on other data, so data_dir is a test set: all of its subjects.
    save_checkpoint(checkpoint_dir / "best.pt", small_config(tmp_path / "train_data"))
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(
            {
                "geometry": "geometry.yaml",
                "checkpoint_dir": str(checkpoint_dir),
                "data": {"data_dir": str(tmp_path / "train_data"), "heldout_views": ["la_2ch"]},
                "evaluate": {
                    "data_dir": str(data_dir),
                    "conditioning_views": ["all", "sa"],
                    "n_samples": 2,
                    "save_predictions": True,
                },
            },
            f,
        )
    evaluate.main(["--config", str(config_path), "--device", "cpu", "--num_workers", "0"])
    out_dir = checkpoint_dir / "evaluation_best"
    rows = read_rows(out_dir)
    assert [(row["conditioning"], row["subject"]) for row in rows] == [("all", subject.name), ("sa", subject.name)]
    assert (out_dir / "all" / subject.name / "la_2ch_sample_std.nii.gz").exists()


def test_direct_can_be_left_out(tmp_path):
    data_dir = tmp_path / "data"
    subject = write_subject(data_dir)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, small_config(data_dir))
    common = ["--checkpoint", str(checkpoint), "--subjects", subject.name, "--device", "cpu", "--num_workers", "0"]

    out_dir = tmp_path / "samples_only"
    evaluate.main([*common, "--out_dir", str(out_dir), "--n_samples", "2", "--no-direct", "--save_predictions"])
    (row,) = read_rows(out_dir)
    assert "mse_direct" not in row and "psnr_direct" not in row
    assert float(row["psnr_sample_mean"]) > 0 and -1 <= float(row["spearman_std_error"]) <= 1
    written = sorted(path.name for path in (out_dir / "all" / subject.name).iterdir())
    assert written == ["la_2ch_mask.nii.gz", "la_2ch_pred_sample_mean.nii.gz", "la_2ch_sample_std.nii.gz", "la_2ch_target.nii.gz"]

    # The same samples as with the direct predictor scored too.
    with_direct = tmp_path / "with_direct"
    evaluate.main([*common, "--out_dir", str(with_direct), "--n_samples", "2"])
    (full,) = read_rows(with_direct)
    assert full["psnr_sample_mean"] == row["psnr_sample_mean"] and "mse_direct" in full

    # Nothing left to score.
    with pytest.raises(SystemExit):
        evaluate.main([*common, "--out_dir", str(tmp_path / "nothing"), "--n_samples", "0", "--no-direct"])
