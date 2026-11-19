r"""Finding each subject's images, and splitting short-axis stacks into slices."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Image


@dataclass
class SubjectFiles:
    path: Path
    saxes: list[Path]  # Every short-axis stack, aligned together.
    la4ch: Path | None = None
    la3ch: Path | None = None
    la2ch: Path | None = None
    seg_sax: Path | None = None
    seg_la4ch: Path | None = None


def find_sax_stacks(subject: Path) -> list[Path]:
    r"""sa_1.nii.gz, sa_2.nii.gz, ... in numerical order."""

    def number(path: Path) -> str:
        return path.name[len("sa_") : -len(".nii.gz")]

    stacks = [path for path in Path(subject).glob("sa_*.nii.gz") if number(path).isdigit()]
    return sorted(stacks, key=lambda path: int(number(path)))


def split_sax_into_slices(sax_nii: Nifti1Image) -> list[Nifti1Image]:
    r"""Each slice of a short-axis stack as its own image, with its own affine."""
    sax_im = sax_nii.get_fdata()
    sax_aff = sax_nii.affine
    slice_niis = []
    for slice_idx in range(sax_im.shape[2]):
        # Offset of this slice from the stack's origin, in scanner space.
        dist_from_vol_origin = sax_aff @ np.array([0, 0, slice_idx, 0])
        slice_affine = sax_aff.copy()
        slice_affine[:3, 3] += dist_from_vol_origin[:3]
        slice_niis.append(nib.Nifti1Image(sax_im[:, :, slice_idx : slice_idx + 1, :], slice_affine))
    return slice_niis


def find_subjects(data_dir, sax_number: int | None = None) -> list[SubjectFiles]:
    r"""Every subject (a numbered directory) with at least one short-axis stack
    and one long-axis image, sorted.

    All of a subject's short-axis stacks are listed, to be aligned in one
    optimisation. `sax_number` restricts it to a single stack -- only for
    debugging, since separate per-stack runs each overwrite the long-axis
    corrections.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{data_dir} is not a directory")
    subject_list = []
    for subject in sorted(data_dir.glob("[0-9]*")):
        if sax_number is None:
            sax_paths = find_sax_stacks(subject)
        else:
            sax_paths = [path for path in [subject / f"sa_{sax_number}.nii.gz"] if path.is_file()]
        if not sax_paths:
            print(f"No sax for subject {subject.name}, skipping")
            continue

        def optional(name):
            path = subject / name
            return path if path.is_file() else None

        files = SubjectFiles(
            path=subject,
            saxes=sax_paths,
            la2ch=optional("la_2ch.nii.gz"),
            la3ch=optional("la_3ch.nii.gz"),
            la4ch=optional("la_4ch.nii.gz"),
            seg_sax=optional("seg_sa_ED.nii.gz"),
            seg_la4ch=optional("seg_la_4ch_ED.nii.gz"),
        )
        if files.la2ch or files.la3ch or files.la4ch:
            subject_list.append(files)
    if not subject_list:
        raise RuntimeError(f"No subject in {data_dir} has both a short-axis stack and a long-axis image")
    return subject_list
