r"""The encoder's input: every available view backprojected onto the latent grid."""

import torch
from torch import Tensor

from .forward_model import BatchOperator


def backproject(
    A: BatchOperator,
    ys: list[list[Tensor]],
    variances: list[list[Tensor]],
    coverage_threshold: float = 1e-2,
    min_variance: float = 1e-6,
) -> Tensor:
    r"""Two channels on the latent grid, for each observation of the batch.

    With W the precision 1 / variance of each view:

    - Channel 0, b = A^T W y / A^T W 1: at each latent voxel, the
      precision-weighted average of the observed voxels that cover it. Zero
      (the normalised data mean) where coverage is below `coverage_threshold`
      of its maximum.
    - Channel 1, c = A^T W 1 / max(A^T W 1): relative coverage, so the encoder
      knows where data are missing, e.g. between short-axis slices or where a
      long-axis view is absent.

    ys:        per observation, per view (in the order of `A.views`), (1, 1, ...).
    variances: the same layout, 0-d tensors.
    Returns:   (B, 2, D, H, W)
    """
    stacked = []
    for views_y, views_var in zip(ys, variances, strict=True):
        stacked.append(
            [
                torch.cat([y, torch.ones_like(y)], dim=1) / var.clamp_min(min_variance)
                for y, var in zip(views_y, views_var, strict=True)
            ]
        )
    # One adjoint for both channels: (B, 2, D, H, W) = [A^T W y, A^T W 1].
    projected = A.adjoint(stacked)
    aty, at1 = projected[:, :1], projected[:, 1:]

    peak = at1.amax(dim=(1, 2, 3, 4), keepdim=True)
    peak = torch.where(peak > 0, peak, torch.ones_like(peak))
    covered = at1 > coverage_threshold * peak
    average = torch.where(
        covered, aty / torch.where(covered, at1, torch.ones_like(at1)), torch.zeros_like(aty)
    )
    coverage = at1.clamp_min(0) / peak
    return torch.cat([average, coverage], dim=1)
