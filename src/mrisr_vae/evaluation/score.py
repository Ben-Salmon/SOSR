r"""Scores every prediction in an evaluation directory, and compares the methods.

    uv run mrisr-score --protocol configs/evaluation/kaggle_cmr_test.yaml
    uv run mrisr-score --out_dir <dir> [--methods vae supervised]

Every <view>_pred_<method>.nii.gz -- a model's from `mrisr-evaluate`, a
baseline's from `mrisr-baselines project`, or any other method's saved there
on the same grid -- is scored by the same code (`metrics`: mse, psnr, ssim)
against the target and mask beside it. Where a <view>_std_<method>.nii.gz is
there too, also `mean_std` (its mean over the mask) and `spearman_std_error`
(its rank correlation with the absolute error).

Writes, in the directory:

    scores.csv    one row per method, conditioning set, subject and view
    summary.csv   per view, conditioning set and method: mean, std and median
                  over the subjects that every method in that conditioning set
                  has (n_subjects), so the methods are compared on the same
                  subjects; n_scored counts all of the method's subjects.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from ..metrics import masked_metrics, masked_spearman
from .files import write_rows
from .protocol import load_protocol

METRICS = ("mse", "psnr", "ssim")
UNCERTAINTY = ("mean_std", "spearman_std_error")


def _visible(paths):
    return [path for path in paths if not path.name.startswith(".")]


def score_directory(out_dir, methods=None) -> list[dict]:
    r"""A row per prediction under `out_dir` (only `methods`' if given)."""
    rows = []
    for target_path in _visible(sorted(Path(out_dir).glob("*/*/*_target.nii.gz"))):
        view = target_path.name[: -len("_target.nii.gz")]
        directory = target_path.parent
        target = nib.load(str(target_path))
        target_data = target.get_fdata()
        mask = nib.load(str(directory / f"{view}_mask.nii.gz")).get_fdata()
        prefix = f"{view}_pred_"
        for prediction_path in _visible(sorted(directory.glob(f"{prefix}*.nii.gz"))):
            method = prediction_path.name[len(prefix) : -len(".nii.gz")]
            if methods is not None and method not in methods:
                continue
            prediction = nib.load(str(prediction_path))
            if prediction.shape != target.shape or not np.allclose(prediction.affine, target.affine, atol=1e-3):
                raise ValueError(f"{prediction_path} is not on the target's image grid")
            prediction_data = prediction.get_fdata()
            row = {
                "method": method,
                "conditioning": directory.parent.name,
                "subject": directory.name,
                "view": view,
                "frame": int(target.header["toffset"]),
                "n_voxels": int((mask > 0.5).sum()),
                **masked_metrics(prediction_data, target_data, mask),
            }
            std_path = directory / f"{view}_std_{method}.nii.gz"
            if std_path.is_file():
                std = nib.load(str(std_path)).get_fdata()
                row["mean_std"] = float(std[mask > 0.5].mean())
                row["spearman_std_error"] = masked_spearman(std, np.abs(prediction_data - target_data), mask)
            rows.append(row)
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    r"""Per view, conditioning set and method, statistics over the subjects
    every method of that view and conditioning set was scored on."""
    summary = []
    groups = sorted({(row["view"], row["conditioning"]) for row in rows})
    for view, conditioning in groups:
        group = [row for row in rows if row["view"] == view and row["conditioning"] == conditioning]
        methods = sorted({row["method"] for row in group})
        subjects = {method: {row["subject"] for row in group if row["method"] == method} for method in methods}
        common = set.intersection(*subjects.values())
        for method in methods:
            selected = [row for row in group if row["method"] == method and row["subject"] in common]
            entry = {
                "view": view,
                "conditioning": conditioning,
                "method": method,
                "n_subjects": len(selected),
                "n_scored": len(subjects[method]),
            }
            for key in METRICS:
                values = np.array([row[key] for row in selected], dtype=np.float64)
                values = values[np.isfinite(values)]
                entry[f"{key}_mean"] = float(values.mean()) if values.size else float("nan")
                entry[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else float("nan")
                entry[f"{key}_median"] = float(np.median(values)) if values.size else float("nan")
            for key in UNCERTAINTY:
                values = np.array([row[key] for row in selected if key in row], dtype=np.float64)
                if values.size:
                    entry[f"{key}_mean"] = float(np.nanmean(values))
            summary.append(entry)
    return summary


def print_summary(summary: list[dict]):
    if not summary:
        print("Nothing scored")
        return
    print(f"{'view':8} {'conditioning':14} {'method':14} {'n':>5}  {'mse':>10} {'psnr':>13} {'ssim':>13}")
    for e in summary:
        print(
            f"{e['view']:8} {e['conditioning']:14} {e['method']:14} {e['n_subjects']:>5}  "
            f"{e['mse_mean']:10.4g} {e['psnr_mean']:7.2f}±{e['psnr_std']:<5.2f} {e['ssim_mean']:7.4f}±{e['ssim_std']:<5.3f}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--protocol", help="An evaluation protocol yaml: scores its out_dir")
    where.add_argument("--out_dir", help="An evaluation directory")
    parser.add_argument("--methods", nargs="+", help="Only these methods. Default: every prediction found")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir if args.out_dir is not None else load_protocol(args.protocol).out_dir)

    rows = score_directory(out_dir, methods=args.methods)
    if not rows:
        print(f"No predictions in {out_dir}")
        return
    summary = summarise(rows)
    write_rows(out_dir / "scores.csv", rows)
    write_rows(out_dir / "summary.csv", summary)
    print_summary(summary)
    print(f"Wrote {out_dir / 'scores.csv'} and {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
