r"""Scoring an evaluation directory, and the checks that keep every method on
the same voxels. CPU only, no model."""

import csv

import nibabel as nib
import numpy as np
import pytest
import torch

from mrisr_vae.evaluation import files, score


def item(subject="s1", frame=2, shape=(16, 16, 1)):
    r"""What the file helpers read of an item: one held-out view, uncropped."""
    return {
        "subject": subject,
        "frame": frame,
        "views": {"la_4ch": {"image_shape": shape, "crop_start": [0, 0, 0], "image_affine": torch.eye(4)}},
    }


def write(out_dir, conditioning, it, target, mask, predictions):
    directory = files.subject_dir(out_dir, conditioning, it["subject"])
    files.write_or_verify_references(directory, it, "la_4ch", target, mask)
    for method, values in predictions.items():
        files.save_on_image_grid(directory, it, "la_4ch", values, files.prediction_name(method))


def test_summary_compares_methods_on_their_common_subjects(tmp_path):
    rng = np.random.default_rng(0)
    target = torch.from_numpy(100 + 10 * rng.standard_normal((16, 16, 1)))
    mask = torch.ones(16, 16, 1)
    good, bad = target + 1.0, target + 3.0
    # Method a has both subjects, b only s1; c is only under another conditioning set.
    write(tmp_path, "all", item("s1"), target, mask, {"a": good, "b": good})
    write(tmp_path, "all", item("s2"), target, mask, {"a": bad})
    write(tmp_path, "sa_first", item("s1"), target, mask, {"c": bad})
    # A half-written file of a running job is not read.
    (tmp_path / "all" / "s1" / ".tmp-1-la_4ch_pred_z.nii.gz").write_bytes(b"")

    rows = score.score_directory(tmp_path)
    assert sorted((r["conditioning"], r["method"], r["subject"]) for r in rows) == [
        ("all", "a", "s1"), ("all", "a", "s2"), ("all", "b", "s1"), ("sa_first", "c", "s1"),
    ]
    assert all(r["frame"] == 2 and r["n_voxels"] == 256 for r in rows)
    summary = {(e["conditioning"], e["method"]): e for e in score.summarise(rows)}
    # Within `all`, on s1 alone, which both a and b have.
    assert summary[("all", "a")]["n_subjects"] == 1 and summary[("all", "a")]["n_scored"] == 2
    assert summary[("all", "a")]["mse_mean"] == pytest.approx(1.0, rel=1e-3)
    assert summary[("all", "b")]["mse_mean"] == pytest.approx(1.0, rel=1e-3)
    assert summary[("sa_first", "c")]["mse_mean"] == pytest.approx(9.0, rel=1e-3)

    score.main(["--out_dir", str(tmp_path), "--methods", "a"])
    assert {r["method"] for r in _read(tmp_path / "scores.csv")} == {"a"}


def test_another_protocols_target_or_mask_is_refused(tmp_path):
    target = torch.full((16, 16, 1), 100.0)
    mask = torch.ones(16, 16, 1)
    directory = files.subject_dir(tmp_path, "all", "s1")
    files.write_or_verify_references(directory, item(), "la_4ch", target, mask)
    # The same again, to float error: accepted.
    files.write_or_verify_references(directory, item(), "la_4ch", target + 1e-4, mask)
    other_mask = mask.clone()
    other_mask[0] = 0
    for it, t, m in ((item(), target, other_mask), (item(), target + 1, mask), (item(frame=3), target, mask)):
        with pytest.raises(RuntimeError):
            files.write_or_verify_references(directory, it, "la_4ch", t, m)


def test_a_prediction_off_the_target_grid_is_refused(tmp_path):
    target = torch.full((16, 16, 1), 100.0)
    write(tmp_path, "all", item(), target, torch.ones(16, 16, 1), {})
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 1), np.float32), np.eye(4)), str(tmp_path / "all" / "s1" / "la_4ch_pred_x.nii.gz"))
    with pytest.raises(ValueError):
        score.score_directory(tmp_path)


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))
