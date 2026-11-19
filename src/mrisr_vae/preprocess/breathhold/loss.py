import torch

# The loss of a pair with no in-bounds sample on its intersection line.
NO_OVERLAP_LOSS = 9999.0


def masked_l2(sample1: torch.Tensor, sample2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""Mean squared difference per pair (batch dimension 0) over `mask`.
    Arrays are (pairs, points, frames)."""
    diff = sample1 - sample2
    metric = diff * diff * mask
    metric = torch.sum(metric, dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8)
    metric[~mask.any(2).any(1)] = NO_OVERLAP_LOSS
    return metric
