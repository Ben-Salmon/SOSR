r"""One subject's slices, loaded for the alignment."""

from dataclasses import dataclass
from itertools import combinations

import nibabel as nib
import numpy as np
import torch

from .files import SubjectFiles, split_sax_into_slices
from .geometry import la_plane_valid_for_sa_stack

# All of a subject's intensities are divided by this percentile of its voxels.
INTENSITY_PERCENTILE = 99.0


@dataclass
class OptimizableImage:
    image: np.ndarray
    affine: np.ndarray
    spacing: np.ndarray
    name: str | None = None
    seg: np.ndarray | None = None


class NoValidLongAxisPlanes(Exception):
    """Raised when a subject ends up with no usable long-axis plane -- either
    none were present, or every long-axis-named file turned out to be
    geometrically invalid (e.g. actually short-axis oriented)."""


class SubjectData:
    def __init__(self, subject: SubjectFiles, device="cpu"):
        self.subject_path = subject.path
        self.planes = []

        sa_seg_data = None
        if subject.seg_sax is not None:
            sa_seg_data = nib.load(subject.seg_sax).get_fdata()  # (x, y, z)

        # Every sa stack, split into slices. All the stacks are aligned in this
        # one optimisation so that they share one long-axis pose: aligned in
        # separate runs, each run moved the long-axis planes to suit its own
        # stack and saved them to the same file, so only the last stack's pose
        # survived. Loaded before the long-axis planes so each long-axis
        # candidate can be validated against the short-axis geometry (some
        # long-axis-named files are actually short-axis slices, which would
        # otherwise crash the plane intersection later on).
        stacks = []  # (name, slices, affines, shapes, spacings)
        for sax_path in subject.saxes:
            slices = split_sax_into_slices(nib.load(sax_path))
            stacks.append(
                (
                    sax_path.name.split(".")[0],
                    slices,
                    [torch.tensor(s.affine, dtype=torch.float32) for s in slices],
                    [s.shape[:2] for s in slices],
                    [s.header.get_zooms()[:2] for s in slices],
                )
            )

        # la_3ch and la_2ch have no segmentations.
        la4ch_seg = None
        if subject.seg_la4ch is not None:
            seg_data = nib.load(subject.seg_la4ch).get_fdata()
            while seg_data.ndim > 2:
                seg_data = seg_data.squeeze(-1)
            la4ch_seg = seg_data[:, :, np.newaxis]  # (x, y, 1)

        # For each long-axis plane kept, the stacks it intersects. It is paired
        # only with their slices.
        la_stacks = []
        for file, seg in zip([subject.la4ch, subject.la3ch, subject.la2ch], [la4ch_seg, None, None]):
            if file is None:
                continue
            nii = nib.load(file)
            la_affine = torch.tensor(nii.affine, dtype=torch.float32)
            valid_stacks = {
                name
                for name, _, affines, shapes, spacings in stacks
                if la_plane_valid_for_sa_stack(la_affine, affines, shapes, spacings)
            }
            if not valid_stacks:
                print(
                    f"Subject {subject.path.stem}: {file.name} does not intersect all "
                    "slices of any short-axis stack (likely mislabeled as long-axis) "
                    "- skipping this file"
                )
                continue
            image_data = nii.get_fdata()[:, :, nii.shape[2] // 2][:, :, :1]  # (x, y, 1): first frame only
            self.planes.append(
                OptimizableImage(image_data, nii.affine, nii.header.get_zooms()[:3], file.name.split(".")[0], seg)
            )
            la_stacks.append(valid_stacks)

        if not self.planes:
            raise NoValidLongAxisPlanes(
                f"Subject {subject.path.stem} has no valid long-axis planes "
                "(missing, or every long-axis-named file was geometrically invalid)"
            )

        # A stack no long-axis plane intersects has nothing to be aligned to; it
        # is left uncorrected (see `save_transformations`).
        aligned = set().union(*la_stacks)
        self.unaligned_stacks = {}  # name -> number of slices
        sa_slices = []  # (stack name, slice index, slice image)
        for name, slices, _, _, _ in stacks:
            if name not in aligned:
                print(f"Subject {subject.path.stem}: no long-axis plane intersects {name} - leaving it uncorrected")
                self.unaligned_stacks[name] = len(slices)
                continue
            for slice_idx, slice_nii in enumerate(slices):
                sa_slices.append((name, slice_idx, slice_nii))

        n_la = len(self.planes)
        # Every pair of long-axis planes, and each long-axis plane with every
        # slice of the stacks it intersects.
        la_la_product = [*combinations(range(n_la), 2)]
        sa_la_product = [
            (i, n_la + j) for i in range(n_la) for j, (name, _, _) in enumerate(sa_slices) if name in la_stacks[i]
        ]
        self.idx_pairs = torch.tensor(la_la_product + sa_la_product)

        for name, slice_idx, slice_nii in sa_slices:
            image_data = slice_nii.get_fdata().squeeze(2)[:, :, :1]  # (x, y, 1): first frame only
            # The segmentation is indexed by slice. It is used for any stack it
            # fits, and skipped for one it does not.
            if (
                sa_seg_data is not None
                and sa_seg_data.shape[:2] == image_data.shape[:2]
                and slice_idx < sa_seg_data.shape[2]
            ):
                seg = sa_seg_data[:, :, slice_idx : slice_idx + 1]  # (x, y, 1)
            else:
                seg = None
            self.planes.append(
                OptimizableImage(image_data, slice_nii.affine, slice_nii.header.get_zooms()[:3], name, seg)
            )

        assert min([min(i) for i in self.idx_pairs]) == 0
        assert max([max(i) for i in self.idx_pairs]) == len(self.planes) - 1

        # Intensities scaled to about [0, 1], so that the loss's intensity and
        # segmentation terms have similar sizes. One factor for the whole
        # subject: a factor per view would depend on what each field of view
        # contains, and so make views that agree disagree.
        voxels = np.concatenate([p.image.ravel() for p in self.planes])
        scale = max(float(np.percentile(voxels, INTENSITY_PERCENTILE)), 1e-8)
        images = [torch.tensor(p.image / scale, dtype=torch.float32, device=device) for p in self.planes]
        self.shapes = torch.stack([torch.tensor(i.shape, dtype=torch.int64, device=device) for i in images])
        self.affines = torch.stack(
            [torch.tensor(p.affine, dtype=torch.float32, device=device) for p in self.planes], dim=0
        )
        self.spacings = torch.stack(
            [torch.tensor(p.spacing, dtype=torch.float32, device=device) for p in self.planes], dim=0
        )
        self.names = [p.name for p in self.planes]
        max_im_shape = self.shapes.amax(dim=0)
        self.images_pad = torch.zeros((len(images), *max_im_shape), dtype=torch.float32, device=device)
        for i, (im, sh) in enumerate(zip(images, self.shapes)):
            self.images_pad[i, : sh[0], : sh[1]] = im
        # Binary heart masks; all zero for a plane without a segmentation, which
        # `has_seg` marks.
        self.has_seg = torch.tensor([p.seg is not None for p in self.planes], dtype=torch.bool, device=device)
        segs = [
            torch.tensor(
                np.zeros_like(p.image) if p.seg is None else (p.seg > 0.5), dtype=torch.float32, device=device
            )
            for p in self.planes
        ]
        self.segs_pad = torch.zeros((len(segs), *max_im_shape), dtype=torch.float32, device=device)
        for i, (seg, sh) in enumerate(zip(segs, self.shapes)):
            self.segs_pad[i, : sh[0], : sh[1]] = seg

    def load_voxel_transformations(self) -> tuple[torch.Tensor, list[str]]:
        r"""The corrections saved by `save_transformations`, as (n_planes, 4, 4)
        voxel transforms (corrected affines are affines @ these), and the views
        with no usable file, which are left uncorrected."""
        transforms = torch.eye(4, dtype=self.affines.dtype, device=self.affines.device).repeat(len(self.planes), 1, 1)
        missing, loaded, slice_counts = set(), {}, {}
        for i, plane in enumerate(self.planes):
            name = plane.name
            if name not in loaded:
                path = self.subject_path / f"{name}_affine_correction.npy"
                loaded[name] = np.load(path) if path.is_file() else None
            correction = loaded[name]
            if correction is None:
                missing.add(name)
            elif "sa" not in name:
                transforms[i] = torch.as_tensor(correction, dtype=transforms.dtype)
            else:
                k = slice_counts.get(name, 0)
                slice_counts[name] = k + 1
                if correction.ndim != 2 or k >= len(correction):
                    missing.add(name)  # A file from a stack with a different number of slices.
                else:
                    transforms[i, :2, 3] = torch.as_tensor(correction[k], dtype=transforms.dtype)
        for name in missing:
            transforms[[i for i, p in enumerate(self.planes) if p.name == name]] = torch.eye(4, dtype=transforms.dtype)
        return transforms, sorted(missing)

    def save_transformations(self, transformations: list[torch.Tensor]):
        r"""Writes <subject>/<view>_affine_correction.npy: a (4, 4) voxel
        transform per long-axis image, (n_slices, 2) whole-voxel in-plane
        translations per short-axis stack."""
        sa_slice_translations = {}  # stack name -> per-slice translations
        for i, plane in enumerate(self.planes):
            if "sa" not in plane.name:
                np.save(str(self.subject_path / plane.name) + "_affine_correction.npy", transformations[i].cpu().numpy())
            else:
                sa_slice_translations.setdefault(plane.name, []).append(transformations[i])
        for name, translations in sa_slice_translations.items():
            np.save(str(self.subject_path / name) + "_affine_correction.npy", torch.stack(translations).cpu().numpy())
        # An explicit zero correction for stacks that could not be aligned, so
        # that a file left over from an earlier run is not applied instead.
        for name, n_slices in self.unaligned_stacks.items():
            np.save(str(self.subject_path / name) + "_affine_correction.npy", np.zeros((n_slices, 2), dtype=np.float32))
