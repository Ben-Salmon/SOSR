r"""Registration parameters: where each view sits relative to the latent grid.

    uv run mrisr-register --dataset cmr --data_dirs <train_dir> <test_dir> [--n_jobs 16] [--check]
    uv run mrisr-register --dataset brain --data_dirs <dir>

Writes <subject>/registration_params.pickle, which `CMRObservationDataset`
reads:

    <view>: vox_to_mm_affine (4, 4)   the view's voxels -> mm, from the NIfTI
                                      qform, with any breathhold correction
            mm_per_vox (3,)           voxel size
            kernel_size_mm (3,)       slice profile width (axis 2: thickness)
            size_vox (3,)             image size
            affine_correction         the breathhold correction applied, so the
                                      dataset can tell when the pickle is stale
    mm_to_su_affine (4, 4)            mm -> scanner units (SU), the latent's frame:
                                      a translation putting the ROI's centre of
                                      mass at the origin

Datasets:

    cmr    Every short-axis stack sa_<n> and la_4ch, la_2ch. The ROI is the
           heart's bounding box (`heart_roi`) in the first of those views
           (normally sa_1), after breathhold correction.
    brain  The axial view `ax`. The ROI is the voxels brighter than 50, and the
           affine is rotated to the image axes.

Run after `mrisr-breathhold`, and again whenever it is rerun. With --check,
nothing is written: each subject's parameters are compared with its existing
pickle and the differences listed.
"""

import argparse
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from ..data import translate_slices
from .breathhold.files import find_sax_stacks
from .heart_roi import NoHeartROI, extract_roi_mask

CMR_LA_VIEWS = ("la_4ch", "la_2ch")
BRAIN_VIEW = "ax"
BRAIN_THRESHOLD = 50
PICKLE = "registration_params.pickle"


class SkipSubject(Exception):
    """A subject that cannot be registered; the message says why."""


# ----------------------------------------------------------------------------
# Affines


def apply_affine(affine: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r"""(4, 4) affine applied to (3, N) points."""
    coords = np.concatenate((coords, np.ones_like(coords[:1])), 0)
    return (affine @ coords)[:3]


def quatern_to_matrix(b, c, d) -> np.ndarray:
    a = b**2 + c**2 + d**2
    a = np.sqrt(max(1.0 - a, 0.0))  # handling floating point errors
    R = np.empty((3, 3))
    R[0, 0] = a**2 + b**2 - c**2 - d**2
    R[0, 1] = 2 * (b * c - a * d)
    R[0, 2] = 2 * (b * d + a * c)
    R[1, 0] = 2 * (b * c + a * d)
    R[1, 1] = a**2 + c**2 - b**2 - d**2
    R[1, 2] = 2 * (c * d - a * b)
    R[2, 0] = 2 * (b * d - a * c)
    R[2, 1] = 2 * (c * d + a * b)
    R[2, 2] = a**2 + d**2 - c**2 - b**2
    return R


def qform(header) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""The header's qform: the (4, 4) voxels -> mm affine, its (3, 3)
    rotation, and the (3,) voxel sizes (the last signed by qfac)."""
    scale = np.zeros((3, 3))
    pixdim = header["pixdim"]
    for i in range(3):
        scale[i, i] = pixdim[i + 1]
    scale[2, 2] *= pixdim[0]
    rotation = quatern_to_matrix(header["quatern_b"], header["quatern_c"], header["quatern_d"])
    if np.linalg.det(rotation) < 0:
        raise SkipSubject(f"qform rotation has determinant {np.linalg.det(rotation)}")
    affine = rotation @ scale
    translation = np.array([[header["qoffset_x"]], [header["qoffset_y"]], [header["qoffset_z"]]])
    affine = np.concatenate((affine, translation), axis=1)
    affine = np.concatenate((affine, np.zeros_like(affine[:1])), 0)
    affine[3, 3] = 1
    return affine, rotation, np.diagonal(scale).copy()


def mm_to_su(centre_mm: np.ndarray) -> torch.Tensor:
    r"""The translation taking `centre_mm` (3,) to the origin."""
    affine = np.eye(4)
    affine[:3, 3] = -centre_mm
    return torch.from_numpy(affine)


# ----------------------------------------------------------------------------
# Breathhold corrections


def load_correction(subject: Path, view: str) -> np.ndarray | None:
    path = subject / f"{view}_affine_correction.npy"
    return np.load(path) if path.is_file() else None


def apply_affine_correction(image: nib.Nifti1Image, correction: np.ndarray, translate_data: bool = True):
    r"""The image with a breathhold correction applied: a (4, 4) voxel
    transform of a long-axis image's affine, or (n_slices, 2) whole-voxel
    in-plane translations of a short-axis stack's slices.

    The new image's qform is set from its affine, as `mrisr-register` reads
    it. With `translate_data` off, the slices are not translated, for when
    only the header is needed: translations do not move the affine.
    """
    if correction.shape == (4, 4):
        return nib.Nifti1Image(image.dataobj, affine=image.affine @ correction, header=image.header)
    if correction.ndim == 2 and correction.shape[1] == 2:
        data = translate_slices(image.get_fdata(), correction) if translate_data else image.dataobj
        return nib.Nifti1Image(data, affine=image.affine, header=image.header)
    raise ValueError(f"Unexpected breathhold correction shape {correction.shape}")


# ----------------------------------------------------------------------------
# Datasets


def view_params(image: nib.Nifti1Image, correction: np.ndarray | None = None) -> dict:
    r"""A CMR view's entry in the pickle, from its (corrected) image."""
    vox_to_mm_affine, _, mm_per_vox = qform(image.header)
    params = {}
    if correction is not None:
        params["affine_correction"] = torch.from_numpy(np.asarray(correction, dtype=np.float64))
    params["vox_to_mm_affine"] = torch.from_numpy(vox_to_mm_affine)
    params["mm_per_vox"] = torch.from_numpy(mm_per_vox.copy())
    params["kernel_size_mm"] = torch.from_numpy(mm_per_vox.copy())
    params["size_vox"] = torch.as_tensor(image.shape[:3])
    return params


def cmr_views(subject: Path) -> list[str]:
    r"""The subject's views, in order: its short-axis stacks, then la_4ch, la_2ch."""
    views = [path.name[: -len(".nii.gz")] for path in find_sax_stacks(subject)] + list(CMR_LA_VIEWS)
    return [view for view in views if (subject / f"{view}.nii.gz").exists()]


def cmr_roi_coords_vox(image: nib.Nifti1Image) -> np.ndarray:
    r"""(3, N) voxels of the heart's bounding box."""
    try:
        return np.stack(np.nonzero(extract_roi_mask(image)), axis=0)
    except NoHeartROI as error:
        raise SkipSubject(str(error)) from error


def corrected_image(subject: Path, view: str, translate_data: bool) -> tuple[nib.Nifti1Image, np.ndarray | None]:
    image = nib.load(subject / f"{view}.nii.gz")
    correction = load_correction(subject, view)
    if correction is not None:
        image = apply_affine_correction(image, correction, translate_data=translate_data)
    return image, correction


def register_cmr(subject: Path) -> dict:
    views = cmr_views(subject)
    if not views:
        raise SkipSubject("no view")
    params = {}
    centre_mm = None
    for view in views:
        roi_view = centre_mm is None
        image, correction = corrected_image(subject, view, translate_data=roi_view)
        params[view] = view_params(image, correction)
        if roi_view:
            coords_mm = apply_affine(params[view]["vox_to_mm_affine"].numpy(), cmr_roi_coords_vox(image))
            centre_mm = np.mean(coords_mm, axis=1)
    params["mm_to_su_affine"] = mm_to_su(centre_mm)
    return params


def brain_vox_to_mm(header) -> tuple[np.ndarray, np.ndarray]:
    r"""The qform with its rotation undone, so the image axes are the mm
    axes, and the voxel sizes."""
    affine, rotation, mm_per_vox = qform(header)
    inv_rotation = np.zeros((4, 4))
    inv_rotation[:3, :3] = rotation.T
    inv_rotation[3, 3] = 1
    return inv_rotation @ affine, mm_per_vox


def brain_roi_coords_vox(image: nib.Nifti1Image) -> np.ndarray:
    data = image.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    return np.stack(np.where(data > BRAIN_THRESHOLD), 0)


def register_brain(subject: Path) -> dict:
    path = subject / f"{BRAIN_VIEW}.nii.gz"
    if not path.is_file():
        raise SkipSubject(f"no {path.name}")
    image = nib.load(path)
    vox_to_mm_affine, mm_per_vox = brain_vox_to_mm(image.header)
    coords_mm = apply_affine(vox_to_mm_affine, brain_roi_coords_vox(image))
    if coords_mm.shape[1] == 0:
        raise SkipSubject(f"no voxel brighter than {BRAIN_THRESHOLD}")
    return {
        BRAIN_VIEW: {
            "vox_to_mm_affine": torch.from_numpy(vox_to_mm_affine),
            "mm_per_vox": torch.from_numpy(mm_per_vox.copy()),
            "kernel_size_mm": torch.from_numpy(np.array(image.header["pixdim"][1:4])),
            "size_vox": torch.as_tensor(image.shape[:3]),
        },
        "mm_to_su_affine": mm_to_su(np.mean(coords_mm, axis=1)),
    }


REGISTER = {"cmr": register_cmr, "brain": register_brain}


# ----------------------------------------------------------------------------
# Running


def differences(new: dict, old: dict, prefix: str = "") -> list[str]:
    r"""How two registration parameter dicts differ (empty if identical)."""
    found = []
    for key in sorted(set(new) | set(old), key=str):
        name = f"{prefix}{key}"
        if key not in old:
            found.append(f"{name}: new")
        elif key not in new:
            found.append(f"{name}: missing")
        elif isinstance(new[key], dict) or isinstance(old[key], dict):
            if not (isinstance(new[key], dict) and isinstance(old[key], dict)):
                found.append(f"{name}: one is not a dict")
            else:
                found += differences(new[key], old[key], prefix=f"{name}/")
        else:
            a, b = torch.as_tensor(new[key]), torch.as_tensor(old[key])
            if a.shape != b.shape:
                found.append(f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
            elif not torch.equal(a.to(torch.float64), b.to(torch.float64)):
                diff = float((a.to(torch.float64) - b.to(torch.float64)).abs().max())
                found.append(f"{name}: max abs difference {diff:.3g}")
    return found


def process_subject(dataset: str, subject: Path, check: bool) -> tuple[str, str, list[str]]:
    r"""(subject, status, details): status is written, skipped, same, differs or no_pickle."""
    try:
        params = REGISTER[dataset](subject)
    except SkipSubject as error:
        return subject.name, "skipped", [str(error)]
    path = subject / PICKLE
    if not check:
        with open(path, "wb") as f:
            pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
        return subject.name, "written", []
    if not path.is_file():
        return subject.name, "no_pickle", []
    with open(path, "rb") as f:
        found = differences(params, pickle.load(f))
    return subject.name, "differs" if found else "same", found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(REGISTER))
    parser.add_argument("--data_dirs", nargs="+", required=True, help="Directories of subject directories")
    parser.add_argument("--subjects", nargs="+", help="Only these subject names")
    parser.add_argument("--max_subjects", type=int, help="Only the first N subjects (sorted)")
    parser.add_argument("--n_jobs", type=int, default=16)
    parser.add_argument("--check", action="store_true", help="Write nothing; compare with the existing pickles")
    args = parser.parse_args(argv)

    subjects = sorted(subject for data_dir in args.data_dirs for subject in Path(data_dir).glob("[0-9]*"))
    if args.subjects is not None:
        subjects = [s for s in subjects if s.name in set(args.subjects)]
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_subject)(args.dataset, subject, args.check) for subject in tqdm(subjects, ncols=80)
    )
    by_status = {}
    for name, status, details in results:
        by_status.setdefault(status, []).append((name, details))
    for name, details in by_status.get("differs", []):
        print(f"{name} differs: {'; '.join(details)}")
    for name, details in by_status.get("skipped", []):
        print(f"{name} skipped: {details[0]}")
    if by_status.get("no_pickle"):
        print(f"No existing pickle: {' '.join(name for name, _ in by_status['no_pickle'])}")
    counts = ", ".join(f"{len(v)} {k}" for k, v in sorted(by_status.items()))
    print(f"{len(subjects)} subjects: {counts}")


if __name__ == "__main__":
    main()
