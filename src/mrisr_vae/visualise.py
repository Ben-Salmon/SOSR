r"""Central slices of volumes, as 8-bit images for logging."""

import numpy as np
import torch
from torch import Tensor


def central_slices(x: Tensor) -> dict[str, Tensor]:
    r"""The three central slices of a (D, H, W) volume."""
    return {
        "front": torch.select(x, 0, x.shape[0] // 2),
        "side": torch.select(x, 1, x.shape[1] // 2),
        "top": torch.select(x, 2, x.shape[2] // 2),
    }


def to_uint8(image: Tensor) -> np.ndarray:
    r"""A 2D image scaled between its 1st and 99th percentiles, ignoring a
    2-pixel border, which often holds edge artefacts."""
    image = image.detach().float().cpu().numpy()
    interior = image[2:-2, 2:-2] if min(image.shape) > 4 else image
    lower, upper = np.percentile(interior, [1, 99])
    scaled = (image - lower) / max(upper - lower, 1e-8)
    return np.rint(np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def volume_images(volumes: dict[str, Tensor]) -> dict[str, np.ndarray]:
    r"""`{name}/{front|side|top}` -> 8-bit image, for (D, H, W) volumes."""
    images = {}
    for name, volume in volumes.items():
        for axis, image in central_slices(volume).items():
            images[f"{name}/{axis}"] = to_uint8(image)
    return images
