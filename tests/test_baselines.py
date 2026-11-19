r"""Interpolation and SMORE baselines, on a synthetic subject. CPU only.
They use each subject's lowest-numbered short-axis stack only.

SMORE itself is not run: its output is faked on a grid twice as fine through
the slices, with the affine SMORE would write, to test how it is read.
"""

import csv
import json

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from mrisr_vae.data import load_frame, translate_slices
from mrisr_vae.evaluation import baselines
from mrisr_vae.evaluation.conditioning import heldout_projection, lvox_to_stack_vox
from mrisr_vae.evaluation.protocol import load_protocol
from mrisr_vae.evaluation.score import score_directory
from mrisr_vae.forward_model import apply_affine

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


def write_protocol(tmp_path, views=("sa_1",), **options):
    r"""A protocol for the baselines alone: la_4ch held out, scored inside sa_first."""
    geometry_path = tmp_path / "geometry.yaml"
    with open(geometry_path, "w") as f:
        yaml.safe_dump(dict(GEOMETRY_CONFIG, VIEWS=list(views)), f)
    protocol = {
        "out_dir": str(tmp_path / "evaluation"),
        "data_dir": str(tmp_path / "data"),
        "geometry": str(geometry_path),
        "heldout_views": ["la_4ch"],
        "region": "sa_first",
        "conditioning_views": ["sa_first"],
        "baselines": {"work_dir": str(tmp_path / "smore")},
        **options,
    }
    path = tmp_path / "protocol.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(protocol, f)
    return str(path)


def run(command, protocol_path, *options):
    baselines.main([command, "--protocol", protocol_path, "--device", "cpu", *options])


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


def test_export_writes_the_evaluated_corrected_frame(tmp_path):
    correction = np.zeros((9, 2), dtype=np.int16)
    correction[2] = [1, -2]
    correction[5] = [-1, 0]
    subject = write_subject(tmp_path / "data", frames=(5, 3), views=VIEWS, sa_correction=correction)
    out_dir = tmp_path / "smore"
    protocol_path = write_protocol(tmp_path)
    run("export", protocol_path)

    dataset = baselines.evaluation_dataset(load_protocol(protocol_path))
    frame = dataset[0]["frame"]
    with open(out_dir / subject.name / "manifest.json") as f:
        assert json.load(f) == {"subject": subject.name, "frame": frame, "stack": "sa_1"}
    img = nib.load(str(subject / "sa_1.nii.gz"))
    exported = nib.load(str(out_dir / subject.name / "sa_1.nii.gz"))
    assert np.array_equal(exported.get_fdata(), translate_slices(load_frame(img, frame), correction))
    assert np.allclose(exported.affine, img.affine)
    assert exported.header.get_zooms() == (1.5, 1.5, 6.0)
    assert (out_dir / "stacks.txt").read_text().split() == [str((out_dir / subject.name / "sa_1.nii.gz").resolve())]


def test_export_writes_only_the_lowest_numbered_stack(tmp_path):
    data_dir = tmp_path / "data"
    write_subject(data_dir, name="both", views=("sa_1", "sa_2", "la_4ch"), frames=(5, 3, 4))
    write_subject(data_dir, name="no_sa_1", views=("sa_2", "la_4ch"), frames=(3, 4))
    out_dir = tmp_path / "smore"
    run("export", write_protocol(tmp_path, views=("sa_1", "sa_2")))
    for name, stack in (("both", "sa_1"), ("no_sa_1", "sa_2")):
        assert sorted(p.name for p in (out_dir / name).glob("*.nii.gz")) == [f"{stack}.nii.gz"]
        with open(out_dir / name / "manifest.json") as f:
            assert json.load(f)["stack"] == stack
    assert len((out_dir / "stacks.txt").read_text().split()) == 2


def test_every_source_recovers_a_ramp_on_the_latent_grid(tmp_path):
    geom = geometry(views=["sa_1"])
    subject = write_ramp_subject(tmp_path / "data")
    out_dir = tmp_path / "smore"
    protocol_path = write_protocol(tmp_path)
    item = baselines.evaluation_dataset(load_protocol(protocol_path))[0]
    run("export", protocol_path)
    fake_smore(out_dir, subject.name)

    vox = apply_affine(lvox_to_stack_vox(item, "sa_1", geom), baselines.latent_coords(geom).T).T
    expected = ramp(*vox.T).reshape(geom.size_lvox)
    # Away from the edges, where cubic taps would be clamped.
    sizes = torch.tensor([36, 36, 9])
    interior = ((vox >= 1) & (vox <= sizes - 3)).all(dim=-1).reshape(geom.size_lvox)
    assert interior.sum() > 1000
    for source in baselines.SOURCES:
        volume = baselines.latent_volume(item, "sa_1", source, out_dir, geom)
        assert torch.allclose(volume[interior], expected[interior], atol=1e-3), source


def test_project_writes_scorable_predictions(tmp_path):
    geom = geometry(views=["sa_1"])
    subject = write_ramp_subject(tmp_path / "data")
    out_dir = tmp_path / "smore"
    protocol_path = write_protocol(tmp_path)
    run("export", protocol_path)
    fake_smore(out_dir, subject.name)
    run("project", protocol_path)

    # The target and mask the models are scored on, written by project itself.
    item = baselines.evaluation_dataset(load_protocol(protocol_path))[0]
    projection = heldout_projection(item, "la_4ch", geom, region="sa_first")
    subject_dir = tmp_path / "evaluation" / "sa_first" / subject.name
    mask = nib.load(str(subject_dir / "la_4ch_mask.nii.gz")).get_fdata()
    assert mask.sum() == projection.mask.sum()
    for source in baselines.SOURCES:
        saved = nib.load(str(subject_dir / f"la_4ch_pred_{source}.nii.gz"))
        assert saved.header["toffset"] == item["frame"]
        volume = baselines.latent_volume(item, "sa_1", source, out_dir, geom)
        expected = projection.project(baselines.normalise(volume, item))
        on_grid = saved.get_fdata()[tuple(slice(s, s + n) for s, n in zip(item["views"]["la_4ch"]["crop_start"],
                                                                         expected.shape))]
        assert np.allclose(on_grid, expected.numpy(), atol=1e-3)
    rows = score_directory(tmp_path / "evaluation")
    assert sorted(row["method"] for row in rows) == sorted(baselines.SOURCES)
    assert all(np.isfinite(row["psnr"]) for row in rows)
    with open(tmp_path / "evaluation" / "runs" / "cubic.csv") as f:
        (row,) = list(csv.DictReader(f))
    assert row["conditioning_views"] == "sa_1" and row["region_stack"] == "sa_1"
    # Run again, it checks the target and mask instead of rewriting them.
    run("project", protocol_path, "--sources", "linear")


def test_project_skips_subjects_without_smore_output(tmp_path, capsys):
    subject = write_ramp_subject(tmp_path / "data")
    protocol_path = write_protocol(tmp_path)
    run("export", protocol_path)
    run("project", protocol_path)
    subject_dir = tmp_path / "evaluation" / "sa_first" / subject.name
    assert (subject_dir / "la_4ch_pred_cubic.nii.gz").is_file()
    assert not (subject_dir / "la_4ch_pred_smore.nii.gz").exists()
    assert "no SMORE output" in capsys.readouterr().out


def test_baselines_need_their_protocol_section(tmp_path):
    write_ramp_subject(tmp_path / "data")
    with pytest.raises(SystemExit):
        run("export", write_protocol(tmp_path, baselines=None))


def test_check_finds_smore_best_aligned_unshifted(tmp_path):
    subject = write_ramp_subject(tmp_path / "data")
    out_dir = tmp_path / "smore"
    protocol_path = write_protocol(tmp_path)
    run("export", protocol_path)
    fake_smore(out_dir, subject.name)
    run("check", protocol_path)

    with open(out_dir / "check.csv") as f:
        rows = list(csv.DictReader(f))
    psnr = {(r["source"], float(r["shift"])): float(r["psnr"]) for r in rows}
    assert psnr[("smore", 0.0)] > psnr[("smore", 0.5)]
    assert psnr[("smore", 0.0)] > psnr[("smore", -0.5)]
    # The fake SMORE output and the interpolations hold the same ramp.
    assert abs(psnr[("smore", 0.0)] - psnr[("linear", 0.0)]) < 1.0

