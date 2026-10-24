r"""Interpolation and SMORE baselines, on a synthetic subject. CPU only.

SMORE itself is not run: its output is faked on a grid twice as fine through
the slices, with the affine SMORE would write, to test how it is read.
"""

import argparse
import csv
import json

import nibabel as nib
import numpy as np
import torch

from mrisr_vae import baselines
from mrisr_vae.data import load_frame, translate_slices
from mrisr_vae.evaluate import heldout_projection, lvox_to_stack_vox, save_predictions
from mrisr_vae.forward_model import apply_affine
from mrisr_vae.metrics import score_directory

from synthetic import GEOMETRY_CONFIG, geometry, write_subject

VIEWS = ("sa_1", "la_4ch")


def ramp(i, j, k):
    r"""A linear function of a stack's voxel indices, which every
    interpolation here reproduces exactly away from the edges."""
    return 3.0 * i - 2.0 * j + 5.0 * k + 200.0


def ramp_volume(shape, z_step=1.0):
    i, j, k = np.meshgrid(*[np.arange(s, dtype=np.float64) for s in shape], indexing="ij")
    return ramp(i, j, k * z_step).astype(np.float32)


def write_ramp_subject(data_dir, sa_correction=None):
    r"""A subject whose sa_1 frames are all the ramp."""
    subject = write_subject(data_dir, frames=(5, 3), views=VIEWS, sa_correction=sa_correction)
    img = nib.load(str(subject / "sa_1.nii.gz"))
    frames = np.repeat(ramp_volume(img.shape[:3])[..., None], img.shape[3], axis=-1)
    nib.save(nib.Nifti1Image(frames, img.affine), str(subject / "sa_1.nii.gz"))
    return subject


def write_checkpoint(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"geometry_config": dict(GEOMETRY_CONFIG, VIEWS=["sa_1"]), "config": {"model": {}}}, path)
    return path


def args_for(data_dir, out_dir, **overrides):
    args = dict(
        data_dir=str(data_dir), out_dir=str(out_dir), view="la_4ch", subjects=None, max_subjects=None, frame=None
    )
    return argparse.Namespace(**{**args, **overrides})


def fake_smore(out_dir, subject, stack="sa_1"):
    r"""SMORE's output for an exported ramp stack: twice as many slices, at
    half the spacing, starting on the first input slice, with the affine
    that says so."""
    exported = nib.load(str(out_dir / subject / f"{stack}.nii.gz"))
    x, y, z = exported.shape
    affine = exported.affine @ np.diag([1.0, 1.0, 0.5, 1.0])
    path = baselines.smore_path(out_dir, subject, stack)
    path.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(ramp_volume((x, y, 2 * z - 1), z_step=0.5), affine), str(path))


def test_sample_reproduces_polynomials_inside_the_volume():
    generator = torch.Generator().manual_seed(0)
    shape = (7, 8, 9)
    grid = torch.stack(torch.meshgrid(*[torch.arange(s, dtype=torch.float64) for s in shape], indexing="ij"))
    points = 1 + torch.rand((50, 3), generator=generator, dtype=torch.float64) * (torch.tensor(shape) - 4)

    def linear(c):
        return 2 * c[0] - c[1] + 0.5 * c[2] + 3

    def quadratic(c):
        return c[0] ** 2 - c[1] * c[2] + c[2]

    for order in (1, 3):
        assert torch.allclose(baselines.sample(linear(grid), points, order), linear(points.T))
    assert torch.allclose(baselines.sample(quadratic(grid), points, 3), quadratic(points.T))


def test_sample_takes_the_edge_value_outside_the_volume():
    volume = torch.arange(4, dtype=torch.float64)[None, None, :].expand(3, 3, 4).contiguous()
    points = torch.tensor([[1.0, 1.0, -5.0], [1.0, 1.0, 9.0], [-2.0, 7.0, 1.0]])
    for order in (1, 3):
        assert torch.allclose(baselines.sample(volume, points, order), torch.tensor([0.0, 3.0, 1.0], dtype=torch.float64))


def test_fuse_averages_covering_stacks_and_falls_back_to_one():
    values = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 4.0, 5.0])]
    inside = [torch.tensor([True, True, False]), torch.tensor([True, False, False])]
    assert torch.equal(baselines.fuse(values, inside, fallback=1).float(), torch.tensor([2.0, 2.0, 5.0]))


def test_export_writes_the_evaluated_corrected_frame(tmp_path):
    correction = np.zeros((9, 2), dtype=np.int16)
    correction[2] = [1, -2]
    correction[5] = [-1, 0]
    subject = write_subject(tmp_path / "data", frames=(5, 3), views=VIEWS, sa_correction=correction)
    out_dir = tmp_path / "smore"
    baselines.main(
        ["export", "--checkpoint", str(write_checkpoint(tmp_path)), "--data_dir", str(tmp_path / "data"),
         "--out_dir", str(out_dir), "--device", "cpu"]
    )

    dataset, _ = baselines.evaluation_dataset(args_for(tmp_path / "data", out_dir), geometry(views=["sa_1"]))
    frame = dataset[0]["frame"]
    with open(out_dir / subject.name / "manifest.json") as f:
        assert json.load(f) == {"subject": subject.name, "frame": frame, "stacks": ["sa_1"]}
    img = nib.load(str(subject / "sa_1.nii.gz"))
    exported = nib.load(str(out_dir / subject.name / "sa_1.nii.gz"))
    assert np.array_equal(exported.get_fdata(), translate_slices(load_frame(img, frame), correction))
    assert np.allclose(exported.affine, img.affine)
    assert exported.header.get_zooms() == (1.5, 1.5, 6.0)
    assert (out_dir / "stacks.txt").read_text().split() == [str((out_dir / subject.name / "sa_1.nii.gz").resolve())]


def test_every_source_recovers_a_ramp_on_the_latent_grid(tmp_path):
    geom = geometry(views=["sa_1"])
    subject = write_ramp_subject(tmp_path / "data")
    out_dir = tmp_path / "smore"
    args = args_for(tmp_path / "data", out_dir)
    dataset, _ = baselines.evaluation_dataset(args, geom)
    item = dataset[0]
    baselines.export(args, geom)
    fake_smore(out_dir, subject.name)

    vox = apply_affine(lvox_to_stack_vox(item, "sa_1", geom), baselines.latent_coords(geom).T).T
    expected = ramp(*vox.T).reshape(geom.size_lvox)
    # Away from the edges, where cubic taps would be clamped.
    sizes = torch.tensor([36, 36, 9])
    interior = ((vox >= 1) & (vox <= sizes - 3)).all(dim=-1).reshape(geom.size_lvox)
    assert interior.sum() > 1000
    for source in baselines.SOURCES:
        volume = baselines.latent_volume(item, ["sa_1"], source, out_dir, geom)
        assert torch.allclose(volume[interior], expected[interior], atol=1e-3), source


def test_project_writes_scorable_predictions(tmp_path):
    geom = geometry(views=["sa_1"])
    subject = write_ramp_subject(tmp_path / "data")
    out_dir = tmp_path / "smore"
    checkpoint = write_checkpoint(tmp_path)
    common = ["--checkpoint", str(checkpoint), "--data_dir", str(tmp_path / "data"), "--out_dir", str(out_dir),
              "--device", "cpu"]
    baselines.main(["export", *common])
    fake_smore(out_dir, subject.name)

    # What mrisr-evaluate --save_predictions --conditioning_views sa writes.
    evaluation_dir = tmp_path / "evaluation"
    dataset, _ = baselines.evaluation_dataset(args_for(tmp_path / "data", out_dir), geom)
    item = dataset[0]
    projection = heldout_projection(item, "la_4ch", geom, region="sa")
    subject_dir = evaluation_dir / "sa" / subject.name
    save_predictions(str(subject_dir), item, "la_4ch", {}, projection.target, projection.mask)

    baselines.main(["project", *common, "--evaluation_dir", str(evaluation_dir)])
    for source in baselines.SOURCES:
        saved = nib.load(str(subject_dir / f"la_4ch_pred_{source}.nii.gz"))
        assert saved.header["toffset"] == item["frame"]
        volume = baselines.latent_volume(item, ["sa_1"], source, out_dir, geom)
        expected = projection.project(baselines.normalise(volume, item))
        on_grid = saved.get_fdata()[tuple(slice(s, s + n) for s, n in zip(item["views"]["la_4ch"]["crop_start"],
                                                                         expected.shape))]
        assert np.allclose(on_grid, expected.numpy(), atol=1e-3)
        rows = score_directory(evaluation_dir, source)
        assert len(rows) == 1 and np.isfinite(rows[0]["psnr"])


def test_project_skips_subjects_without_smore_output(tmp_path, capsys):
    geom = geometry(views=["sa_1"])
    subject = write_ramp_subject(tmp_path / "data")
    out_dir = tmp_path / "smore"
    common = ["--checkpoint", str(write_checkpoint(tmp_path)), "--data_dir", str(tmp_path / "data"),
              "--out_dir", str(out_dir), "--device", "cpu"]
    baselines.main(["export", *common])
    evaluation_dir = tmp_path / "evaluation"
    dataset, _ = baselines.evaluation_dataset(args_for(tmp_path / "data", out_dir), geom)
    item = dataset[0]
    projection = heldout_projection(item, "la_4ch", geom, region="sa")
    save_predictions(str(evaluation_dir / "sa" / subject.name), item, "la_4ch", {}, projection.target, projection.mask)

    baselines.main(["project", *common, "--evaluation_dir", str(evaluation_dir)])
    assert (evaluation_dir / "sa" / subject.name / "la_4ch_pred_cubic.nii.gz").is_file()
    assert not (evaluation_dir / "sa" / subject.name / "la_4ch_pred_smore.nii.gz").exists()
    assert "no SMORE output" in capsys.readouterr().out


def test_check_finds_smore_best_aligned_unshifted(tmp_path):
    subject = write_ramp_subject(tmp_path / "data")
    out_dir = tmp_path / "smore"
    common = ["--checkpoint", str(write_checkpoint(tmp_path)), "--data_dir", str(tmp_path / "data"),
              "--out_dir", str(out_dir), "--device", "cpu"]
    baselines.main(["export", *common])
    fake_smore(out_dir, subject.name)
    baselines.main(["check", *common])

    with open(out_dir / "check.csv") as f:
        rows = list(csv.DictReader(f))
    psnr = {(r["source"], float(r["shift"])): float(r["psnr"]) for r in rows}
    assert psnr[("smore", 0.0)] > psnr[("smore", 0.5)]
    assert psnr[("smore", 0.0)] > psnr[("smore", -0.5)]
    # The fake SMORE output and the interpolations hold the same ramp.
    assert abs(psnr[("smore", 0.0)] - psnr[("linear", 0.0)]) < 1.0

