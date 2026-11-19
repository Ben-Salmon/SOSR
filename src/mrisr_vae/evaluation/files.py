r"""The evaluation directory: one per protocol, shared by every method.

    <out_dir>/protocol.yaml                       the protocol's shared settings
    <out_dir>/<conditioning>/<subject>/<view>_target.nii.gz
                                       <view>_mask.nii.gz         1 on the scored voxels
                                       <view>_pred_<method>.nii.gz
                                       <view>_std_<method>.nii.gz the projected samples' std
    <out_dir>/runs/<method>.csv                   what each method's run did, per subject
    <out_dir>/scores.csv, summary.csv             `mrisr-score`

Every file is on the held-out view's own image grid (its NIfTI affine and
shape), in raw intensities, for the evaluated frame (header toffset).
Whichever job comes first writes the target and mask; every later one
recomputes them and refuses to go on if they differ, so all methods are scored
on the same frame and voxels.
"""

import csv
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

REFERENCE_TOLERANCE = {"target": 1e-2, "mask": 1e-6}


def subject_dir(out_dir, conditioning: str, subject: str) -> Path:
    return Path(out_dir) / conditioning / subject


def prediction_name(method: str) -> str:
    return f"pred_{method}"


def std_name(method: str) -> str:
    return f"std_{method}"


def on_image_grid(values: torch.Tensor, heldout: dict) -> np.ndarray:
    r"""Places values on the view's cropped grid into its full image, zero elsewhere."""
    full = np.zeros(heldout["image_shape"], dtype=np.float32)
    x0, y0, z0 = heldout["crop_start"]
    n0, n1, n2 = values.shape
    full[x0 : x0 + n0, y0 : y0 + n1, z0 : z0 + n2] = values.detach().cpu().numpy()
    return full


def save_atomically(nifti: nib.Nifti1Image, path: Path):
    r"""Saves via a temporary file, so a concurrent reader never sees half a file."""
    path = Path(path)
    tmp = path.with_name(f".tmp-{os.getpid()}-{path.name}")
    nib.save(nifti, str(tmp))
    os.replace(tmp, path)


def image(item: dict, view: str, values: torch.Tensor) -> nib.Nifti1Image:
    r"""Values on `view`'s cropped grid as a NIfTI on its image grid, with the
    frame in the header's toffset."""
    heldout = item["views"][view]
    nifti = nib.Nifti1Image(on_image_grid(values, heldout), affine=heldout["image_affine"].numpy())
    if item["frame"] is not None:
        nifti.header["toffset"] = item["frame"]
    return nifti


def save_on_image_grid(directory, item: dict, view: str, values: torch.Tensor, name: str):
    r"""Writes <directory>/<view>_<name>.nii.gz (see `image`)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_atomically(image(item, view, values), directory / f"{view}_{name}.nii.gz")


def write_or_verify_references(directory, item: dict, view: str, target: torch.Tensor, mask: torch.Tensor):
    r"""Writes the target and mask, or, if another method already did, checks
    they are the ones it scored against (same frame, values and voxels)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, values in (("target", target), ("mask", mask)):
        path = directory / f"{view}_{name}.nii.gz"
        new = image(item, view, values)
        if not path.is_file():
            save_atomically(new, path)
            continue
        saved = nib.load(str(path))
        same_frame = float(saved.header["toffset"]) == float(new.header["toffset"])
        same_grid = saved.shape == new.shape and np.allclose(saved.affine, new.affine, atol=1e-4)
        if not (same_frame and same_grid and np.allclose(saved.get_fdata(), new.get_fdata(), atol=REFERENCE_TOLERANCE[name])):
            raise RuntimeError(
                f"{path} differs from this run's {name}: another protocol (data, frame or region) "
                "wrote this directory. Use a fresh out_dir."
            )


def write_rows(path, rows: list[dict]):
    r"""A CSV with every key of every row as a column."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
