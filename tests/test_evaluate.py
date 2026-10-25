r"""Held-out-view evaluation, end to end on a synthetic subject. CPU only.

Here the model is mostly trained on sa_1 only and la_2ch plays the held-out
view; the scored-region tests hold out la_4ch, as the CMR configs do.
"""

import copy
import csv

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml
from nibabel.affines import apply_affine

from mrisr_vae import evaluate
from mrisr_vae.data import CMRObservationDataset
from mrisr_vae.evaluate import norm_views_for, region_stack, resolve_conditioning, stack_height_mm, stack_region
from mrisr_vae.metrics import score_directory
from mrisr_vae.model import build_model
from mrisr_vae.train import DEFAULTS, check_heldout_views, check_relative_std

from synthetic import GEOMETRY_CONFIG, VIEW_GEOMETRY, geometry, make_item, write_subject


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
    # Each subject picks one of these: see test_sa_first_*.
    assert resolve_conditioning("sa_first", views) == ["sa_1", "sa_2"]
    for spec in ("la_4ch", "ax_first", "sa_first,la_2ch"):
        with pytest.raises(ValueError):
            resolve_conditioning(spec, views)


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


def heldout_z_mm(item, geom, view="la_4ch"):
    r"""z (mm; mm_to_su is the identity) of each voxel centre of `view`'s crop."""
    centres = item["views"][view]["op"]["y_centres_lvox"].to(torch.float64)
    affine = geom.lvox_to_su_affine.to(torch.float64)
    return (centres @ affine[:3, :3].T + affine[:3, 3])[..., 2]


def test_stack_region_is_the_stack_through_plane_extent():
    geom = geometry(views=["sa_2", "la_2ch"])
    item = make_item(geom, views=("sa_2", "la_4ch"))
    z = heldout_z_mm(item, geom)
    region = stack_region(item, "sa_2", "la_4ch", geom)
    assert region.shape == item["views"]["la_4ch"]["op"]["row_mask"].shape
    # Slices at z = -14, -8, -2 mm, 6 mm thick: from -17 to 1 mm, the gaps
    # between slices included. In-plane the stack covers the whole latent.
    assert torch.equal(region, (z >= -17) & (z <= 1))
    assert region.any() and not region.all()
    # Half the slice thickness beyond the end slices, not half the spacing.
    item["views"]["sa_2"]["geometry"]["kernel_size_mm"] = torch.tensor([1.5, 1.5, 9.0], dtype=torch.float64)
    assert torch.equal(stack_region(item, "sa_2", "la_4ch", geom), (z >= -18.5) & (z <= 2.5))


def test_region_stack_is_the_tallest_the_latest_breaking_ties():
    geom = geometry(views=["sa_1", "sa_2", "la_2ch"])
    item = make_item(geom, views=("sa_1", "sa_2", "la_2ch"))
    assert stack_height_mm(item["views"]["sa_1"]["geometry"]) == pytest.approx(54.0)  # 9 slices of 6 mm
    assert stack_height_mm(item["views"]["sa_2"]["geometry"]) == pytest.approx(18.0)
    assert region_stack(item, "sa", geom.views) == "sa_1"
    # Only the views given are candidates, e.g. the ones the model was trained on.
    assert region_stack(item, "sa", ["sa_2", "la_2ch"]) == "sa_2"
    assert region_stack(item, "sa", ["la_2ch"]) is None
    # Equal heights: the highest index, compared as a number.
    for view in ("sa_3", "sa_10"):
        item["views"][view] = copy.deepcopy(item["views"]["sa_2"])
    assert region_stack(item, "sa", ["sa_2", "sa_3"]) == "sa_3"
    assert region_stack(item, "sa", ["sa_2", "sa_3", "sa_10"]) == "sa_10"
    # sa_first: the lowest number, compared as a number, whatever the heights.
    assert region_stack(item, "sa_first", geom.views) == "sa_1"
    assert region_stack(item, "sa_first", ["sa_3", "sa_10", "la_2ch"]) == "sa_3"
    assert region_stack(item, "sa_first", ["la_2ch"]) is None


def test_sa_first_normalises_from_the_lowest_numbered_stack_present(tmp_path):
    geom = geometry(views=["sa_1", "sa_2", "la_2ch"])
    write_subject(tmp_path / "both", views=("sa_1", "sa_2", "la_2ch"), frames=(5, 3, 4))
    write_subject(tmp_path / "no_sa_1", views=("sa_2", "la_2ch"), frames=(3, 4))
    norm_views = norm_views_for("sa_first", resolve_conditioning("sa_first", geom.views))
    for data_dir, stack in ((tmp_path / "both", "sa_1"), (tmp_path / "no_sa_1", "sa_2")):
        item = CMRObservationDataset(data_dir, geom, norm_views=norm_views, frame=1)[0]
        alone = CMRObservationDataset(data_dir, geom, norm_views=[stack], frame=1)[0]
        assert item["norm_views"] == [stack]
        assert (item["norm_mean"], item["norm_std"]) == (alone["norm_mean"], alone["norm_std"])
        assert item["norm_mean"] != CMRObservationDataset(data_dir, geom, frame=1)[0]["norm_mean"]


def test_evaluate_conditions_and_scores_on_the_lowest_numbered_stack(tmp_path):
    data_dir = tmp_path / "data"
    # sa_1 covers only the lower part of the latent; sa_2, all of it, so it
    # is the taller one.
    subject = write_subject(
        data_dir, views=("sa_1", "sa_2", "la_2ch", "la_4ch"), frames=(5, 3, 4, 4),
        view_geometry={"sa_1": VIEW_GEOMETRY["sa_2"], "sa_2": VIEW_GEOMETRY["sa_1"]},
    )
    checkpoint = tmp_path / "model.pt"
    config = small_config(data_dir, heldout_views=("la_4ch",))
    geometry_config = dict(GEOMETRY_CONFIG, VIEWS=["sa_1", "sa_2", "la_2ch"])
    torch.manual_seed(0)
    model = build_model(config, geometry(views=geometry_config["VIEWS"]))
    torch.save({"model": model.state_dict(), "config": config, "geometry_config": geometry_config}, checkpoint)

    def run(name, region):
        out_dir = tmp_path / name
        evaluate.main([
            "--checkpoint", str(checkpoint), "--subjects", subject.name, "--conditioning_views", "sa_first", "sa",
            "--n_samples", "2", "--region", region, "--device", "cpu", "--num_workers", "0",
            "--out_dir", str(out_dir),
        ])
        return {row["conditioning"]: row for row in read_rows(out_dir)}

    first = run("first", "sa_first")
    assert first["sa_first"]["conditioning_views"] == "sa_1"
    assert first["sa"]["conditioning_views"] == "sa_1 sa_2"
    assert float(first["sa_first"]["norm_mean"]) != pytest.approx(float(first["sa"]["norm_mean"]))
    # The same region for both sets: sa_1, though sa_2 is taller.
    assert {row["region_stack"] for row in first.values()} == {"sa_1"}
    assert first["sa_first"]["n_voxels"] == first["sa"]["n_voxels"]
    tallest = run("tallest", "sa")
    assert {row["region_stack"] for row in tallest.values()} == {"sa_2"}
    assert int(tallest["sa"]["n_voxels"]) > int(first["sa"]["n_voxels"])


def test_evaluate_scores_only_inside_the_tallest_shortaxis_stack(tmp_path):
    data_dir = tmp_path / "data"
    # Trained on sa_1 and la_2ch, la_4ch held out; sa_1 covers only the
    # lower part of the latent, from z = -17 to 1 mm.
    subject = write_subject(
        data_dir, views=("sa_1", "la_2ch", "la_4ch"), frames=(5, 3, 4), view_geometry={"sa_1": VIEW_GEOMETRY["sa_2"]}
    )
    checkpoint = tmp_path / "model.pt"
    config = small_config(data_dir, heldout_views=("la_4ch",))
    torch.manual_seed(0)
    model = build_model(config, geometry())
    torch.save({"model": model.state_dict(), "config": config, "geometry_config": dict(GEOMETRY_CONFIG)}, checkpoint)
    common = [
        "--checkpoint", str(checkpoint), "--subjects", subject.name, "--conditioning_views", "all", "sa", "la_2ch",
        "--n_samples", "2", "--save_predictions", "--device", "cpu", "--num_workers", "0",
    ]

    def run(name, *options):
        out_dir = tmp_path / name
        evaluate.main([*common, "--out_dir", str(out_dir), *options])
        return out_dir, read_rows(out_dir)

    def mask(out_dir, conditioning):
        return nib.load(str(out_dir / conditioning / subject.name / "la_4ch_mask.nii.gz")).get_fdata() > 0

    whole_dir, whole = run("whole")
    region_dir, region = run("region", "--region", "sa")
    assert [row["conditioning"] for row in region] == ["all", "sa", "la_2ch"]
    assert all(row["region_stack"] == "sa_1" for row in region)
    assert "region_stack" not in whole[0]
    # The same voxels for every conditioning set, even la_2ch alone, which
    # does not see sa_1: fewer than on the whole latent, all among them.
    n_voxels = {int(row["n_voxels"]) for row in region}
    assert len(n_voxels) == 1 and 0 < min(n_voxels) < int(whole[0]["n_voxels"])
    scored = mask(region_dir, "all")
    assert scored.sum() == min(n_voxels)
    assert np.array_equal(scored, mask(region_dir, "sa")) and np.array_equal(scored, mask(region_dir, "la_2ch"))
    whole_mask = mask(whole_dir, "all")
    assert not np.any(scored & ~whole_mask)
    # In mm: the scored voxels lie within the stack; the whole latent reaches above it.
    affine = nib.load(str(subject / "la_4ch.nii.gz")).affine
    z = apply_affine(affine, np.argwhere(scored))[:, 2]
    assert z.min() >= -17 and z.max() <= 1
    assert apply_affine(affine, np.argwhere(whole_mask))[:, 2].max() > 1
    # mrisr-score, on the saved mask, gives the same scores.
    scores = [s for s in score_directory(region_dir, "sample_mean") if s["conditioning"] == "all"]
    assert scores[0]["mse"] == pytest.approx(float(region[0]["mse_sample_mean"]), rel=1e-3)
    assert scores[0]["psnr"] == pytest.approx(float(region[0]["psnr_sample_mean"]), rel=1e-3)

    # none: the whole latent again.
    _, none = run("none", "--region", "none")
    assert none[0]["n_voxels"] == whole[0]["n_voxels"] and none[0]["mse_sample"] == whole[0]["mse_sample"]
    # A group of no view the model was trained on.
    with pytest.raises(SystemExit):
        run("bad", "--region", "ax")
