r"""Small synthetic observations, after diffusion-priors/tests/test_fixes.py."""

import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from mrisr_vae.data import view_geometry as data_view_geometry
from mrisr_vae.forward_model import BatchOperator, build_view_operator
from mrisr_vae.geometry import LatentGeometry

# A 20^3 latent of 2 mm voxels centred on the origin.
GEOMETRY_CONFIG = {
    "VIEWS": ["sa_1", "la_2ch"],
    "SCALE_FACTOR": 1.0,
    "RAW_SU_PER_LVOX": 2.0,
    "x_size_su": [40, 40, 40],
    "X_MIN_COORDS_SU": [-20, -20, -20],
}

# view: (shape, spacing, axes, centre). A short-axis stack covering the latent
# and an oblique single-slice long axis.
VIEW_GEOMETRY = {
    "sa_1": ((36, 36, 9), [1.5, 1.5, 6.0], None, [0.0, 0.0, 0.0]),
    "la_2ch": ((36, 36, 1), [1.5, 1.5, 6.0], (30, 20, 10), [1.0, -2.0, 0.5]),
    # A stack over the lower part of the latent only: slices at z = -14, -8,
    # -2 mm, so from z = -17 to 1 mm with 6 mm slices.
    "sa_2": ((36, 36, 3), [1.5, 1.5, 6.0], None, [0.0, 0.0, -8.0]),
    # A long-axis slice whose second image axis runs along z, through the
    # whole latent, as la_4ch runs from apex to atria: z = (j - 17.5) * 1.5 mm.
    "la_4ch": ((36, 36, 1), [1.5, 1.5, 6.0], (90, 0, 15), [0.5, 1.0, 0.0]),
}


def geometry(views=None) -> LatentGeometry:
    config = dict(GEOMETRY_CONFIG)
    if views is not None:
        config["VIEWS"] = list(views)
    return LatentGeometry.from_config(config)


def rotation(degrees_x, degrees_y, degrees_z):
    ax, ay, az = np.radians([degrees_x, degrees_y, degrees_z])
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def grid_affine(shape, spacing, axes, centre):
    r"""vox -> mm affine of a grid with axes `axes` (columns), centred on `centre`."""
    affine = np.eye(4)
    affine[:3, :3] = np.asarray(axes, dtype=float) @ np.diag(spacing)
    affine[:3, 3] = np.asarray(centre, dtype=float) - affine[:3, :3] @ ((np.asarray(shape[:3]) - 1) / 2)
    return affine


def view_params(view, view_geometry=None):
    r"""`view_geometry` overrides VIEW_GEOMETRY, view by view."""
    shape, spacing, angles, centre = {**VIEW_GEOMETRY, **(view_geometry or {})}[view]
    axes = np.eye(3) if angles is None else rotation(*angles)
    return {
        "vox_to_mm_affine": torch.from_numpy(grid_affine(shape, spacing, axes, centre)),
        "mm_per_vox": torch.tensor(spacing, dtype=torch.float64),
        "kernel_size_mm": torch.tensor(spacing, dtype=torch.float64),
        "size_vox": torch.tensor(shape),
    }


def make_observation(views=("sa_1", "la_2ch")) -> dict:
    r"""Registration parameters of an observation, as `build_view_operator` reads them."""
    observation = {"mm_to_su_affine": torch.eye(4, dtype=torch.float64)}
    for view in views:
        observation[view] = view_params(view)
    return observation


def smooth_volume(geom: LatentGeometry, seed: int = 0) -> torch.Tensor:
    r"""A smooth random latent (1, 1, D, H, W)."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((1, 1, *geom.size_lvox), generator=generator)
    kernel = torch.ones(1, 1, 5, 5, 5) / 125
    return torch.nn.functional.conv3d(x, kernel, padding=2) * 5


def make_item(geom: LatentGeometry, views=("sa_1", "la_2ch"), x_true=None, seed=0, sigma=0.1) -> dict:
    r"""A dataset item with synthetic y: A x_true plus noise if `x_true` is
    given, otherwise pure noise."""
    observation = make_observation(views)
    generator = torch.Generator().manual_seed(seed)
    item_views = {}
    for view in views:
        op = build_view_operator(observation, view, geom)
        shape = tuple(op["row_mask"].shape)
        noise = sigma * torch.randn((1, 1, *shape), generator=generator)
        if x_true is None:
            y = torch.randn((1, 1, *shape), generator=generator)
        else:
            with torch.no_grad():
                y = BatchOperator([{view: op}], geom.size_lvox)(x_true)[0][0] + noise
        item_views[view] = {
            "y": y,
            "sigma_meas": sigma,
            "sigma_extra": sigma,
            "op": op,
            "geometry": data_view_geometry(observation[view]),
        }
    return {
        "subject": f"synthetic_{seed}",
        "frame": 0,
        "mm_to_su_affine": observation["mm_to_su_affine"],
        "norm_mean": 0.0,
        "norm_std": 1.0,
        "views": item_views,
    }


def write_subject(
    root: Path,
    name: str = "subject_a",
    frames=(5, 3),
    views=("sa_1", "la_2ch"),
    seed: int = 0,
    sa_correction=None,
    record_correction=True,
    view_geometry=None,
) -> Path:
    r"""A subject directory as the dataset expects it: one NIfTI per view
    (X, Y, Z, T), registration_params.pickle, and a noise level for sa_1.
    `view_geometry` overrides VIEW_GEOMETRY, view by view."""
    subject = Path(root) / name
    subject.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    params = {"mm_to_su_affine": torch.eye(4, dtype=torch.float64)}
    for view, n_frames in zip(views, frames):
        view_param = view_params(view, view_geometry)
        shape = tuple(int(s) for s in view_param["size_vox"])
        data = (100 + 20 * rng.standard_normal((*shape, n_frames))).astype(np.float32)
        nifti = nib.Nifti1Image(data, affine=view_param["vox_to_mm_affine"].numpy())
        nib.save(nifti, str(subject / f"{view}.nii.gz"))
        params[view] = {k: v for k, v in view_param.items() if k != "size_vox"}
    torch.save(torch.tensor(2.0), subject / "sa_1_noise_std.pt")
    if sa_correction is not None:
        np.save(subject / "sa_1_affine_correction.npy", sa_correction)
        if record_correction:
            params["sa_1"]["affine_correction"] = sa_correction
    with open(subject / "registration_params.pickle", "wb") as f:
        pickle.dump(params, f)
    return subject
