r"""Held-out-view evaluation by protocol, end to end on synthetic subjects. CPU only.

Here the model is mostly trained on sa_1 only and la_2ch plays the held-out
view; the scored-region tests hold out la_4ch, as the CMR protocol does.
"""

import copy
import csv

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml
from nibabel.affines import apply_affine

from mrisr_vae.data import CMRObservationDataset
from mrisr_vae.evaluation import heldout, score
from mrisr_vae.evaluation.conditioning import (
    check_region,
    norm_views_for,
    region_stack,
    resolve_conditioning,
    stack_region,
)
from mrisr_vae.evaluation.protocol import load_protocol
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


def save_checkpoint(path, config, views=("sa_1",)):
    torch.manual_seed(0)
    model = build_model(config, geometry(views=list(views)))
    geometry_config = dict(GEOMETRY_CONFIG, VIEWS=list(views))
    torch.save({"model": model.state_dict(), "config": config, "geometry_config": geometry_config}, path)
    return str(path)


def write_protocol(tmp_path, data_dir, methods, views=("sa_1",), name="protocol", **options):
    r"""A protocol yaml (and its geometry yaml) in tmp_path; returns its path."""
    geometry_path = tmp_path / f"{name}_geometry.yaml"
    with open(geometry_path, "w") as f:
        yaml.safe_dump(dict(GEOMETRY_CONFIG, VIEWS=list(views)), f)
    protocol = {
        "out_dir": str(tmp_path / f"{name}_out"),
        "data_dir": str(data_dir),
        "geometry": str(geometry_path),
        "heldout_views": ["la_2ch"],
        "region": None,
        "conditioning_views": ["all"],
        "methods": methods,
        **options,
    }
    path = tmp_path / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(protocol, f)
    return str(path)


def evaluate(protocol_path, method, *options):
    heldout.main(["--protocol", protocol_path, "--method", method, "--device", "cpu", "--num_workers", "0", *options])


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


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


def test_methods_write_into_one_directory_and_are_scored_together(tmp_path):
    data_dir = tmp_path / "data"
    subject = write_subject(data_dir)
    checkpoint = save_checkpoint(tmp_path / "model.pt", small_config(data_dir))
    # The same checkpoint twice: its posterior samples' mean, and its direct predictor.
    methods = {"vae": {"checkpoint": checkpoint, "n_samples": 2}, "vae_direct": {"checkpoint": checkpoint}}
    protocol_path = write_protocol(tmp_path, data_dir, methods, conditioning_views=["all", "sa"])
    out_dir = tmp_path / "protocol_out"
    evaluate(protocol_path, "vae")
    evaluate(protocol_path, "vae_direct")

    runs = read_csv(out_dir / "runs" / "vae.csv")
    assert [row["conditioning"] for row in runs] == ["all", "sa"]
    assert runs[0]["frame"] == runs[1]["frame"]  # every conditioning set on the same frame
    assert runs[0]["view"] == "la_2ch" and runs[0]["conditioning_views"] == "sa_1"
    assert "mse_sample" not in read_csv(out_dir / "runs" / "vae_direct.csv")[0]

    raw = nib.load(str(subject / "la_2ch.nii.gz"))
    frame = int(runs[0]["frame"])
    subject_dir = out_dir / "all" / subject.name
    written = sorted(path.name for path in subject_dir.iterdir())
    assert written == [
        "la_2ch_mask.nii.gz", "la_2ch_pred_vae.nii.gz", "la_2ch_pred_vae_direct.nii.gz", "la_2ch_std_vae.nii.gz",
        "la_2ch_target.nii.gz",
    ]
    for path in subject_dir.iterdir():
        nifti = nib.load(str(path))
        assert nifti.shape == raw.shape[:3] and np.allclose(nifti.affine, raw.affine), path.name
        assert int(nifti.header["toffset"]) == frame, path.name
    mask = nib.load(str(subject_dir / "la_2ch_mask.nii.gz")).get_fdata() > 0
    assert mask.sum() == int(runs[0]["n_voxels"])
    target = nib.load(str(subject_dir / "la_2ch_target.nii.gz")).get_fdata()
    assert np.allclose(target[mask], raw.get_fdata()[..., frame][mask], atol=1e-2)
    std = nib.load(str(subject_dir / "la_2ch_std_vae.nii.gz")).get_fdata()
    assert np.all(std[~mask] == 0) and np.all(std[mask] >= 0)

    score.main(["--protocol", protocol_path])
    scores = read_csv(out_dir / "scores.csv")
    assert sorted((row["method"], row["conditioning"]) for row in scores) == [
        ("vae", "all"), ("vae", "sa"), ("vae_direct", "all"), ("vae_direct", "sa"),
    ]
    by_key = {(row["method"], row["conditioning"]): row for row in scores}
    vae = by_key[("vae", "all")]
    for key in ("mse", "psnr"):
        assert np.isfinite(float(vae[key])) and np.isfinite(float(by_key[("vae_direct", "all")][key]))
    # The mean of the samples cannot be further from the target than the samples on average.
    assert float(vae["mse"]) <= float(runs[0]["mse_sample"]) * (1 + 1e-6)
    # The projected samples' spread, and its rank correlation with the error; none for the direct predictor.
    assert float(vae["mean_std"]) == pytest.approx(np.mean(std[mask]), rel=1e-4)
    assert -1 <= float(vae["spearman_std_error"]) <= 1
    assert by_key[("vae_direct", "all")]["mean_std"] == ""
    summary = read_csv(out_dir / "summary.csv")
    assert {(row["method"], row["conditioning"], row["n_subjects"]) for row in summary} == {
        ("vae", "all", "1"), ("vae", "sa", "1"), ("vae_direct", "all", "1"), ("vae_direct", "sa", "1"),
    }

    # Another method saved beside them is scored by the same code: a copy of
    # the direct prediction scores as the direct prediction did.
    direct = nib.load(str(subject_dir / "la_2ch_pred_vae_direct.nii.gz"))
    nib.save(direct, str(subject_dir / "la_2ch_pred_copy.nii.gz"))
    (copied,) = [row for row in score.score_directory(out_dir, methods=["copy"])]
    assert copied["mse"] == pytest.approx(float(by_key[("vae_direct", "all")]["mse"]))


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


def test_protocol_is_checked(tmp_path):
    methods = {"vae": {"checkpoint": "model.pt"}}
    load_protocol(write_protocol(tmp_path, tmp_path, methods, name="ok"))
    bad = {
        "trained_view": {"heldout_views": ["sa_1"]},
        "unknown_set": {"conditioning_views": ["la_4ch"]},
        "tallest_region": {"region": "sa"},
        "unknown_option": {"save_predictions": True},
        "baseline_name": {"methods": {"cubic": {"checkpoint": "model.pt"}}},
        "no_checkpoint": {"methods": {"vae": {"n_samples": 2}}},
    }
    for name, options in bad.items():
        with pytest.raises(ValueError):
            load_protocol(write_protocol(tmp_path, tmp_path, options.pop("methods", methods), name=name, **options))
    check_region("sa_first", ["sa_1"])
    with pytest.raises(ValueError):
        check_region("sa", ["sa_1"])


def test_a_method_must_match_the_protocol(tmp_path):
    data_dir = tmp_path / "data"
    write_subject(data_dir)
    checkpoint = save_checkpoint(tmp_path / "model.pt", small_config(data_dir))
    config = small_config(data_dir)
    config["model"]["direct_predictor"] = False
    no_direct = save_checkpoint(tmp_path / "no_direct.pt", config)
    methods = {"ok": {"checkpoint": checkpoint}, "no_direct": {"checkpoint": no_direct}}
    # A checkpoint trained on another grid.
    with pytest.raises(SystemExit):
        evaluate(write_protocol(tmp_path, data_dir, methods, views=("sa_1", "sa_2"), name="other_grid"), "ok")
    protocol_path = write_protocol(tmp_path, data_dir, methods)
    # No direct predictor, and no samples.
    with pytest.raises(SystemExit):
        evaluate(protocol_path, "no_direct")
    with pytest.raises(SystemExit):
        evaluate(protocol_path, "missing")
    # A different protocol cannot write into the same out_dir.
    evaluate(protocol_path, "ok")
    with open(protocol_path) as f:
        changed = yaml.safe_load(f)
    changed["frame"] = 1
    with open(protocol_path, "w") as f:
        yaml.safe_dump(changed, f)
    with pytest.raises(SystemExit):
        evaluate(protocol_path, "ok")


def test_samples_depend_only_on_the_seed_subject_and_conditioning(tmp_path):
    data_dir = tmp_path / "data"
    # subject_a sorts first, so it runs before subject_z when both are evaluated.
    write_subject(data_dir, "subject_z", seed=0)
    write_subject(data_dir, "subject_a", seed=1)
    checkpoint = save_checkpoint(tmp_path / "model.pt", small_config(data_dir))
    methods = {"vae": {"checkpoint": checkpoint, "n_samples": 2}, "direct": {"checkpoint": checkpoint}}

    def run(name, *options, seed=0):
        protocol_path = write_protocol(tmp_path, data_dir, methods, name=name, seed=seed)
        evaluate(protocol_path, "vae", *options)
        evaluate(protocol_path, "direct", *options)
        rows = score.score_directory(tmp_path / f"{name}_out")
        return {(row["method"], row["subject"]): row for row in rows}

    alone = run("alone", "--subjects", "subject_z")
    together = run("together", "--subjects", "subject_z", "subject_a")
    # The same samples for subject_z whether or not subject_a was sampled before it.
    for key in ("mse", "mean_std", "spearman_std_error"):
        assert alone[("vae", "subject_z")][key] == together[("vae", "subject_z")][key], key
    other_seed = run("other_seed", "--subjects", "subject_z", seed=1)
    assert other_seed[("vae", "subject_z")]["mse"] != alone[("vae", "subject_z")]["mse"]
    # The direct predictor is deterministic.
    assert other_seed[("direct", "subject_z")]["mse"] == alone[("direct", "subject_z")]["mse"]


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


def test_region_stack_is_the_lowest_numbered():
    geom = geometry(views=["sa_1", "sa_2", "la_2ch"])
    item = make_item(geom, views=("sa_1", "sa_2", "la_2ch"))
    for view in ("sa_3", "sa_10"):
        item["views"][view] = copy.deepcopy(item["views"]["sa_2"])
    assert region_stack(item, "sa_first", geom.views) == "sa_1"
    # Only the views given are candidates, e.g. the ones the model was trained
    # on; numbers compare as numbers.
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
    # sa_1 covers only the lower part of the latent; sa_2, all of it.
    subject = write_subject(
        data_dir, views=("sa_1", "sa_2", "la_2ch", "la_4ch"), frames=(5, 3, 4, 4),
        view_geometry={"sa_1": VIEW_GEOMETRY["sa_2"], "sa_2": VIEW_GEOMETRY["sa_1"]},
    )
    views = ("sa_1", "sa_2", "la_2ch")
    checkpoint = save_checkpoint(tmp_path / "model.pt", small_config(data_dir, heldout_views=("la_4ch",)), views)
    protocol_path = write_protocol(
        tmp_path, data_dir, {"vae": {"checkpoint": checkpoint, "n_samples": 2}}, views=views,
        heldout_views=["la_4ch"], region="sa_first", conditioning_views=["sa_first", "sa"],
    )
    evaluate(protocol_path, "vae", "--subjects", subject.name)
    runs = {row["conditioning"]: row for row in read_csv(tmp_path / "protocol_out" / "runs" / "vae.csv")}
    assert runs["sa_first"]["conditioning_views"] == "sa_1"
    assert runs["sa"]["conditioning_views"] == "sa_1 sa_2"
    assert float(runs["sa_first"]["norm_mean"]) != pytest.approx(float(runs["sa"]["norm_mean"]))
    # The same region for both sets: sa_1, though sa_2 covers more.
    assert {row["region_stack"] for row in runs.values()} == {"sa_1"}
    assert runs["sa_first"]["n_voxels"] == runs["sa"]["n_voxels"]


def test_evaluate_scores_only_inside_the_first_shortaxis_stack(tmp_path):
    data_dir = tmp_path / "data"
    # Trained on sa_1 and la_2ch, la_4ch held out; sa_1 covers only the
    # lower part of the latent, from z = -17 to 1 mm.
    subject = write_subject(
        data_dir, views=("sa_1", "la_2ch", "la_4ch"), frames=(5, 3, 4), view_geometry={"sa_1": VIEW_GEOMETRY["sa_2"]}
    )
    views = ("sa_1", "la_2ch")
    checkpoint = save_checkpoint(tmp_path / "model.pt", small_config(data_dir, heldout_views=("la_4ch",)), views)
    methods = {"vae": {"checkpoint": checkpoint, "n_samples": 2}}
    options = dict(views=views, heldout_views=["la_4ch"], conditioning_views=["all", "sa", "la_2ch"])

    def run(name, region):
        evaluate(write_protocol(tmp_path, data_dir, methods, name=name, region=region, **options), "vae")
        return tmp_path / f"{name}_out", read_csv(tmp_path / f"{name}_out" / "runs" / "vae.csv")

    def mask(out_dir, conditioning):
        return nib.load(str(out_dir / conditioning / subject.name / "la_4ch_mask.nii.gz")).get_fdata() > 0

    whole_dir, whole = run("whole", None)
    region_dir, region = run("region", "sa_first")
    assert [row["conditioning"] for row in region] == ["all", "sa", "la_2ch"]
    assert all(row["region_stack"] == "sa_1" for row in region)
    assert whole[0]["region_stack"] == ""
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
