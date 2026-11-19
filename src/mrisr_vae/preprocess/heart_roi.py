r"""The 2-D heart ROI bounding box of Section 3.1 of Zambal et al. (2008),
"Fast and robust localization of the heart in cardiac MRI series", as a dense
binary mask. The heart moves over the cardiac cycle, so it is where the
intensity varies most over time.
"""

import nibabel as nib
import numpy as np
from scipy import ndimage


class NoHeartROI(RuntimeError):
    """Raised when no ROI can be found in an image."""


def clean_slice(mask: np.ndarray) -> np.ndarray:
    """The paper's 5x5 rule: fewer than 11 neighbours on -> off, more than 15 -> on."""
    counts = ndimage.convolve(mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), mode="constant", cval=0)
    out = mask.copy()
    out[counts < 11] = False
    out[counts > 15] = True
    return out


def extract_roi_mask(
    img: nib.Nifti1Image, target_area_mm2: float = 8000.0, min_slice_fraction: float = 0.5
) -> np.ndarray:
    r"""(X, Y, Z) uint8 mask, 1 inside the heart's in-plane bounding box on
    every slice, for an (X, Y, Z, T) image.

    Per slice, keeps the `target_area_mm2` of pixels whose intensity varies
    most over the cardiac cycle (cleaned with `clean_slice`); keeps the
    pixels kept in at least `min_slice_fraction` of the slices; and boxes
    the largest 8-connected component of them.
    """
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 4:
        raise NoHeartROI(f"Expected a 4-D image (X, Y, Z, T), not shape {data.shape}")

    variance = np.var(data, axis=3)  # (X, Y, Z)

    sx, sy = img.header.get_zooms()[:2]
    n_keep = int(round(target_area_mm2 / (sx * sy)))
    n_keep = max(1, min(n_keep, variance.shape[0] * variance.shape[1]))

    slice_masks = np.zeros_like(variance, dtype=bool)
    for z in range(variance.shape[2]):
        v = variance[:, :, z]
        threshold = np.partition(v.ravel(), -n_keep)[-n_keep]
        slice_masks[:, :, z] = clean_slice(v >= threshold)

    min_slices = int(np.ceil(min_slice_fraction * variance.shape[2]))
    combined = np.sum(slice_masks, axis=2) >= min_slices

    labels, n_labels = ndimage.label(combined, structure=np.ones((3, 3)))
    if n_labels == 0:
        raise NoHeartROI("No ROI component was found")
    sizes = ndimage.sum(combined, labels, index=np.arange(1, n_labels + 1))
    largest = labels == (1 + int(np.argmax(sizes)))

    xs, ys = np.where(largest)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    roi = np.zeros(data.shape[:3], dtype=np.uint8)
    roi[x0:x1, y0:y1, :] = 1
    return roi
