import torch

# The loss of a pair with too few comparable points on its intersection line.
NO_OVERLAP_LOSS = 9999.0
# Fewer compared points than this counts as no overlap, so that a pair cannot
# score well by matching only a handful of points.
MIN_POINTS = 5
METRICS = ("ncc", "l2")


def intersection_loss(
    im1: torch.Tensor,
    im2: torch.Tensor,
    heart1: torch.Tensor,
    heart2: torch.Tensor,
    mask: torch.Tensor,
    both_segmented: torch.Tensor,
    seg_weight: float,
    metric: str = "ncc",
) -> torch.Tensor:
    r"""Per pair: an intensity dissimilarity over `mask`, plus `seg_weight`
    times the mean squared difference of the heart masks where both planes are
    segmented (`both_segmented`, (pairs,)). The heart term penalises the hearts
    not overlapping, which the intensities alone, compared only inside the
    hearts, do not. Arrays are (pairs, points, frames).

    metric: "ncc", 1 - normalised cross-correlation, which ignores a gain and
    offset between the two profiles, as between slices from separate series;
    or "l2", the mean squared difference, which needs intensities on one scale.
    """
    m = mask.to(im1.dtype)
    n = m.sum(dim=(1, 2))
    n1 = n.clamp(min=1)
    if metric == "l2":
        intensity = ((im1 - im2) ** 2 * m).sum(dim=(1, 2)) / n1
    elif metric == "ncc":
        mean1 = (im1 * m).sum(dim=(1, 2)) / n1
        mean2 = (im2 * m).sum(dim=(1, 2)) / n1
        a = (im1 - mean1[:, None, None]) * m
        b = (im2 - mean2[:, None, None]) * m
        norm = torch.sqrt((a * a).sum(dim=(1, 2)) * (b * b).sum(dim=(1, 2))).clamp(min=1e-8)
        intensity = 1 - (a * b).sum(dim=(1, 2)) / norm
    else:
        raise ValueError(f"metric must be one of {METRICS}, not {metric!r}")
    heart = ((heart1 - heart2) ** 2 * m).sum(dim=(1, 2)) / n1
    loss = intensity + seg_weight * heart * both_segmented
    return torch.where(n < MIN_POINTS, torch.full_like(loss, NO_OVERLAP_LOSS), loss)
