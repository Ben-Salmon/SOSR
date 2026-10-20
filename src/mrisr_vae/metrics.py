r"""Masked image metrics for scoring predictions of a held-out view.

    uv run mrisr-score --evaluation_dir <dir> --name baseline

Every method should be scored by this code on the same voxels: the scoring
mask written by `mrisr-evaluate --save_predictions` (the voxels the held-out
view's forward model supports). Predictions are compared in raw intensities.

- mse:  mean squared error over the mask.
- psnr: 10 log10(L^2 / mse), with L the target's range (max - min) over the
        mask, so the same for every method.
- spearman (std vs error): Spearman's rank correlation, over the mask, of a
        per-voxel uncertainty (the posterior samples' std) and the absolute
        error of the prediction; computed per subject.
- ssim: SSIM (Wang et al., 2004) with an 11 x 11 Gaussian window (sigma 1.5)
        and K1 = 0.01, K2 = 0.03, computed in-plane slice by slice (the third
        image axis is the slice axis), with dynamic range L, averaged over the
        windows that lie entirely inside the mask. Predictions are only defined
        inside the mask, so a window reaching outside it would compare against
        nothing.

`mrisr-score` scores files saved next to `mrisr-evaluate`'s: in
<evaluation_dir>/<conditioning>/<subject>/, for each <view>_target.nii.gz and
<view>_mask.nii.gz there, the prediction <view>_pred_<name>.nii.gz. Save a
baseline's predictions there (on the same image grid, for the frame in the
target's header `toffset`) to score it with exactly the same code.
"""

import argparse
import csv
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03


def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]  # (1, 1, size, size)


def _as_slices(x) -> torch.Tensor:
    r"""(X, Y, Z) -> (Z, 1, X, Y) in float64."""
    x = torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x).to(torch.float64).cpu()
    if x.ndim == 2:
        x = x[..., None]
    return x.permute(2, 0, 1)[:, None]


def masked_ssim(prediction, target, mask, data_range: float) -> float:
    r"""Mean SSIM over the in-plane windows lying entirely inside `mask`.
    Arrays are (X, Y, Z) or (X, Y). NaN if no window fits."""
    p, t = _as_slices(prediction), _as_slices(target)
    m = _as_slices(mask) > 0.5
    if min(p.shape[-2:]) < SSIM_WINDOW:
        return math.nan
    window = _gaussian_window(SSIM_WINDOW, SSIM_SIGMA)

    def filt(a):
        return F.conv2d(a, window)  # valid: windows inside the image

    mu_p, mu_t = filt(p), filt(t)
    var_p = filt(p * p) - mu_p**2
    var_t = filt(t * t) - mu_t**2
    cov = filt(p * t) - mu_p * mu_t
    c1 = (SSIM_K1 * data_range) ** 2
    c2 = (SSIM_K2 * data_range) ** 2
    ssim_map = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / ((mu_p**2 + mu_t**2 + c1) * (var_p + var_t + c2))
    inside = F.conv2d(m.to(torch.float64), torch.ones(1, 1, SSIM_WINDOW, SSIM_WINDOW, dtype=torch.float64))
    inside = inside > SSIM_WINDOW**2 - 0.5
    if not inside.any():
        return math.nan
    return float(ssim_map[inside].mean())


def masked_metrics(prediction, target, mask) -> dict:
    r"""mse, psnr and ssim of `prediction` against `target` on `mask` (> 0.5),
    and the dynamic range L used. Arrays are (X, Y, Z) or (X, Y)."""
    p = torch.as_tensor(np.asarray(prediction) if not isinstance(prediction, torch.Tensor) else prediction)
    t = torch.as_tensor(np.asarray(target) if not isinstance(target, torch.Tensor) else target)
    m = torch.as_tensor(np.asarray(mask) if not isinstance(mask, torch.Tensor) else mask) > 0.5
    p, t, m = p.to(torch.float64).cpu(), t.to(torch.float64).cpu(), m.cpu()
    if not m.any():
        return {"mse": math.nan, "psnr": math.nan, "ssim": math.nan, "data_range": math.nan}
    mse = float(torch.mean((p[m] - t[m]) ** 2))
    data_range = float(t[m].max() - t[m].min())
    psnr = 10 * math.log10(data_range**2 / mse) if mse > 0 and data_range > 0 else math.inf
    ssim = masked_ssim(p, t, m, data_range) if data_range > 0 else math.nan
    return {"mse": mse, "psnr": psnr, "ssim": ssim, "data_range": data_range}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    r"""Ranks 0..n-1, tied values sharing their mean rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.bincount(inverse, weights=ranks) / counts)[inverse]


def masked_spearman(a, b, mask) -> float:
    r"""Spearman's rank correlation of `a` and `b` over `mask` (> 0.5), e.g.
    of the posterior standard deviation and the absolute error per voxel.
    NaN with fewer than 3 voxels, or if either is constant on the mask."""
    m = np.asarray(torch.as_tensor(mask).cpu()) > 0.5
    x = np.asarray(torch.as_tensor(a).to(torch.float64).cpu())[m]
    y = np.asarray(torch.as_tensor(b).to(torch.float64).cpu())[m]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    return float(np.corrcoef(_average_ranks(x), _average_ranks(y))[0, 1])


def score_directory(evaluation_dir, name: str) -> list[dict]:
    r"""Scores every <view>_pred_<name>.nii.gz under `evaluation_dir` against
    the target and mask saved beside it."""
    rows = []
    for target_path in sorted(Path(evaluation_dir).glob("*/*/*_target.nii.gz")):
        view = target_path.name[: -len("_target.nii.gz")]
        subject_dir = target_path.parent
        prediction_path = subject_dir / f"{view}_pred_{name}.nii.gz"
        if not prediction_path.exists():
            print(f"Missing {prediction_path}")
            continue
        target = nib.load(str(target_path))
        prediction = nib.load(str(prediction_path))
        if prediction.shape != target.shape or not np.allclose(prediction.affine, target.affine, atol=1e-3):
            raise ValueError(f"{prediction_path} is not on the target's image grid")
        mask = nib.load(str(subject_dir / f"{view}_mask.nii.gz")).get_fdata()
        metrics = masked_metrics(prediction.get_fdata(), target.get_fdata(), mask)
        rows.append(
            {
                "conditioning": subject_dir.parent.name,
                "subject": subject_dir.name,
                "view": view,
                "frame": float(target.header["toffset"]),
                **metrics,
            }
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluation_dir", required=True, help="An mrisr-evaluate --save_predictions out_dir")
    parser.add_argument("--name", required=True, help="Scores <view>_pred_<name>.nii.gz, e.g. baseline, direct")
    parser.add_argument("--out", help="Default: <evaluation_dir>/scores_<name>.csv")
    args = parser.parse_args(argv)
    rows = score_directory(args.evaluation_dir, args.name)
    if not rows:
        print("Nothing to score")
        return
    out = args.out or str(Path(args.evaluation_dir) / f"scores_{args.name}.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for conditioning in sorted({row["conditioning"] for row in rows}):
        selected = [row for row in rows if row["conditioning"] == conditioning]
        means = ", ".join(f"{key} {np.nanmean([row[key] for row in selected]):.4g}" for key in ("mse", "psnr", "ssim"))
        print(f"{conditioning}: {len(selected)} predictions; mean {means}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
