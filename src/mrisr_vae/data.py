r"""Low-resolution CMR observations, one frame of one subject per item.

Merges diffusion-priors' `LowResDataset.__getitem__` (data.py),
`gather_observation_views` and `crop_y` (utils.py), and adds what the VAE
needs: each view's forward operator, built here so that DataLoader workers
build them in parallel, and random view dropout for training.
"""

import pickle
import random
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, random_split

from .forward_model import apply_affine, bound_y_coords, build_view_operator, invert_affine
from .geometry import LatentGeometry

# Views are normalised jointly to this standard deviation inside the latent,
# as in diffusion-priors.
# TODO: This wasn't necessary. If retraining, remove
NORMALISED_STD = 0.8


def discover_subjects(data_dir: Path, views) -> list[Path]:
    r"""Subjects with registration parameters and at least one view, sorted:
    the same list `LowResDataset` builds."""
    subjects = sorted(Path(data_dir).glob("*"))
    return [
        subject
        for subject in subjects
        if (subject / "registration_params.pickle").exists()
        and any((subject / f"{view}.nii.gz").exists() for view in views)
    ]


def split_subjects(data_dir: Path, views, seed: int = 42) -> tuple[list[str], list[str]]:
    r"""Training and validation subject names.

    The same 90/10 split as diffusion-priors' `utils.build_datasets` with the
    same seed, so both models are validated on the same subjects.
    """
    subjects = discover_subjects(data_dir, views)
    generator = torch.Generator().manual_seed(seed)
    train_split, val_split = random_split(range(len(subjects)), [0.9, 0.1], generator=generator)
    return (
        [subjects[i].stem for i in train_split.indices],
        [subjects[i].stem for i in val_split.indices],
    )


def subjects_with_views(data_dir, names: list[str], views) -> list[str]:
    r"""The subjects of `names` that have every view of `views`, e.g. the
    supervised baseline's target. Subjects where a view exists but cannot be
    used are left out later, by the dataset."""
    return [name for name in names if all((Path(data_dir) / name / f"{view}.nii.gz").is_file() for view in views)]


def subject_rng(subject: str) -> random.Random:
    r"""A random source fixed by the subject, so that a validation subject gets
    the same frame every time, the same one diffusion-priors picks. `zlib.crc32`
    rather than `hash`, which Python salts per process."""
    return random.Random(zlib.crc32(subject.encode()))


def crop_y(
    y: Tensor,
    mm_to_su_affine: Tensor,
    vox_to_mm_affine: Tensor,
    geometry: LatentGeometry,
) -> tuple[Tensor, Tensor, list[int]]:
    """
    Returns the y-voxels that the forward model predicts, together with a
    boolean mask selecting the voxels whose centres lie inside the (rotated)
    latent cube, and the index of the crop's first voxel in y.

    y: (C, X, Y, Z). Returns (C, X', Y', Z'), (X', Y', Z') and [x0, y0, z0].
    """
    vox_to_su_affine = mm_to_su_affine @ vox_to_mm_affine
    su_to_vox_affine = invert_affine(vox_to_su_affine)

    x_coords_su_flat = torch.flatten(geometry.coords_su, start_dim=1, end_dim=-1)
    x_coords_vox = apply_affine(affine=su_to_vox_affine, coords=x_coords_su_flat)
    y_size_vox = torch.as_tensor(y.shape[1:4], dtype=x_coords_vox.dtype)

    y_max_coords_vox, y_min_coords_vox = bound_y_coords(x_coords_vox, y_size_vox)
    slices = [slice(None, None)] + [
        slice(int(y_min_coords_vox[d]), int(y_max_coords_vox[d])) for d in range(3)
    ]
    cropped_y = y[tuple(slices)]

    # Build voxel-centre coords for the cropped region and map them back to SU.
    # Inside the rotated cube  <=>  SU coord lies inside the axis-aligned SU AABB.
    dtype = vox_to_su_affine.dtype
    axes = [
        torch.arange(int(y_min_coords_vox[d]), int(y_max_coords_vox[d]), dtype=dtype)
        for d in range(3)
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)  # [3, X, Y, Z]
    flat = grid.reshape(3, -1)
    su = apply_affine(affine=vox_to_su_affine, coords=flat)  # [3, N]
    lo = geometry.min_coords_su.to(dtype)[:, None]
    hi = geometry.max_coords_su.to(dtype)[:, None]
    inside = ((su >= lo) & (su < hi)).all(dim=0)
    mask = inside.reshape(grid.shape[1:])  # [X, Y, Z]
    start = [int(y_min_coords_vox[d]) for d in range(3)]
    return cropped_y, mask, start


def view_geometry(params: dict) -> dict:
    r"""A view's acquired grid as the forward model sees it: `vox_to_mm_affine`
    (with any rigid correction), `mm_per_vox`, `kernel_size_mm` (the slice
    thickness on axis 2) and the uncropped `size_vox`, as float64 tensors."""
    keys = ("vox_to_mm_affine", "mm_per_vox", "kernel_size_mm", "size_vox")
    return {key: torch.as_tensor(params[key]).to(torch.float64) for key in keys}


class MeanStdTracker:
    def __init__(self):
        self.running_mean = 0
        self.running_sqdev = 0
        self.N = 0

    def update_stats(self, y, mask=None):
        # Chan's parallel Welford: combine running stats with a batch in one shot.
        vals = y[mask] if mask is not None else y.reshape(-1)
        n_b = vals.numel()
        if n_b == 0:
            return
        vals = vals.to(torch.float64)
        mean_b = vals.mean()
        m2_b = ((vals - mean_b) ** 2).sum()
        if self.N == 0:
            self.running_mean = mean_b
            self.running_sqdev = m2_b
            self.N = n_b
            return
        delta = mean_b - self.running_mean
        new_N = self.N + n_b
        self.running_mean = self.running_mean + delta * n_b / new_N
        self.running_sqdev = self.running_sqdev + m2_b + delta**2 * self.N * n_b / new_N
        self.N = new_N

    def get_final_stats(self):
        var = self.running_sqdev / (self.N - 1)
        std = var**0.5
        return float(self.running_mean), float(std)


def translate_slices(frame: np.ndarray, correction: np.ndarray) -> np.ndarray:
    r"""Per-slice in-plane translation of a short-axis frame (X, Y, Z) by whole
    voxels: the (Z, 2) branch of diffusion-priors' `apply_affine_correction`,
    applied to one frame instead of the whole series."""
    X, Y, Z = frame.shape[:3]
    translated = np.zeros_like(frame)
    correction = correction.astype(np.int16)

    for z in range(Z):
        dx, dy = correction[z]

        x_src_start = max(0, -dx)
        x_src_end = min(X, X - dx)
        y_src_start = max(0, -dy)
        y_src_end = min(Y, Y - dy)

        x_dst_start = max(0, dx)
        x_dst_end = x_dst_start + (x_src_end - x_src_start)
        y_dst_start = max(0, dy)
        y_dst_end = y_dst_start + (y_src_end - y_src_start)

        translated[x_dst_start:x_dst_end, y_dst_start:y_dst_end, z] = frame[
            x_src_start:x_src_end, y_src_start:y_src_end, z
        ]
    return translated


def load_frame(img: nib.Nifti1Image, frame: int | None) -> np.ndarray:
    r"""One frame (X, Y, Z) of a 3D or 4D image, reading no other frame into memory."""
    if len(img.shape) == 4:
        data = img.dataobj[:, :, :, frame]
    else:
        data = np.asanyarray(img.dataobj)
    return np.asarray(data, dtype=np.float32)


class CMRObservationDataset(Dataset):
    r"""One frame of one subject's views, cropped to the latent, normalised,
    and each paired with its forward operator.

    Each item is a dict, or None if the subject has no usable view:

        subject, frame, mm_to_su_affine, norm_mean, norm_std,
        views: {view: {y, sigma_meas, sigma_extra, op, input,
                       crop_start, image_shape, image_affine}} in `views` order

    where `y` is (1, 1, n_y0, n_y1, n_y2), `op` the `build_view_operator`
    tensors, and the noise standard deviations are in normalised units.
    `crop_start` places y in the view's image, of shape `image_shape` and with
    the NIfTI file's own affine `image_affine`. `input` is False for a view
    dropped from the encoder's input (see `view_drop_probability`).

    Arguments:
        subjects: Subject names in `data_dir`. Defaults to all usable subjects.
        extra_noise_std: Extra noise standard deviation by view, in the raw
            intensity units of the data, as in diffusion-priors. Unlike there,
            no noise is added to y: it only enters the likelihood.
        views: The views to load. Defaults to the geometry's. Views outside the
            geometry's (held-out views, for evaluation) come after the
            geometry's views; the model never conditions on them.
        norm_views: The views whose voxels set the normalisation. Defaults to
            every input view. Evaluation sets it to the conditioning views,
            so that a held-out view does not leak into the input through its
            intensities.
        train: Random frames and view dropout. Otherwise the frame is fixed per
            subject and every view is kept.
        view_drop_probability: In training, each view is dropped from the
            encoder's input with this probability, keeping at least one. A
            dropped view stays in the likelihood, so the model learns to
            predict views it was not shown; it also does not set the
            normalisation, as at evaluation.
        frame: Use this frame for every subject (subjects without it are skipped).
        target_views: Views to load as targets, e.g. the supervised baseline's:
            loaded after the others, on the same frame, but always `input:
            False`, never chosen by view dropout and never setting the
            normalisation, so nothing of them reaches the encoder. Subjects
            without one are skipped (the item is None).
    """

    def __init__(
        self,
        data_dir,
        geometry: LatentGeometry,
        subjects: list[str] | None = None,
        extra_noise_std: dict | None = None,
        views: list[str] | None = None,
        norm_views: list[str] | None = None,
        train: bool = False,
        view_drop_probability: float = 0.0,
        frame: int | None = None,
        lanczos_a: int = 3,
        target_views: list[str] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.geometry = geometry
        if views is None:
            self.views = list(geometry.views)
        else:
            self.views = [view for view in geometry.views if view in views] + [
                view for view in dict.fromkeys(views) if view not in geometry.views
            ]
        self.target_views = list(dict.fromkeys(target_views or []))
        trained = set(self.target_views) & set(geometry.views)
        if trained:
            raise ValueError(f"Target views {sorted(trained)} are among the views the model is shown")
        self.views += [view for view in self.target_views if view not in self.views]
        self.norm_views = None if norm_views is None else set(norm_views)
        self.extra_noise_std = dict(extra_noise_std or {})
        self.train = train
        self.view_drop_probability = view_drop_probability
        self.frame = frame
        self.lanczos_a = lanczos_a
        self._warned_unrecorded_correction = False
        if subjects is None:
            self.subjects = discover_subjects(self.data_dir, self.views)
        else:
            self.subjects = sorted(self.data_dir / subject for subject in subjects)

    def __len__(self):
        return len(self.subjects)

    def _check_correction(self, subject: str, view: str, params: dict, correction):
        r"""Fails if registration_params.pickle was built with a different
        breathhold correction than the one on disk.

        A long-axis correction only reaches the forward model through the
        affine stored in the pickle, so a pickle built before the corrections
        were last rerun silently uses the old geometry.
        """
        recorded = params.get("affine_correction")
        if recorded is None:
            if not self._warned_unrecorded_correction:
                print(
                    "Warning: registration_params.pickle does not record the "
                    "breathhold corrections it was built with, so it cannot be "
                    "checked against the *_affine_correction.npy files. "
                    "Regenerate it (diffusion-priors/registration/get_registration_params.sh) "
                    "after any rerun of the correction."
                )
                self._warned_unrecorded_correction = True
            return
        recorded = np.asarray(recorded)
        if recorded.shape != correction.shape or not np.allclose(recorded, correction):
            raise RuntimeError(
                f"{subject}/{view}: registration_params.pickle was built with a "
                "different breathhold correction than the one on disk. Rerun "
                "diffusion-priors/registration/get_registration_params.sh."
            )

    def _choose_frame(self, subject: str, images: dict) -> tuple[bool, int | None]:
        r"""A frame every view has: random in training, fixed per subject otherwise."""
        frame_counts = [img.shape[3] for img in images.values() if len(img.shape) == 4]
        if not frame_counts:
            return True, None
        n_frames = min(frame_counts)
        if self.frame is not None:
            if not 0 <= self.frame < n_frames:
                print(f"Subject {subject}: frame {self.frame} is out of range ({n_frames} frames); skipping it")
                return False, None
            return True, self.frame
        if self.train:
            return True, random.randint(0, n_frames - 1)
        return True, subject_rng(subject).randint(0, n_frames - 1)

    def __getitem__(self, idx: int) -> dict | None:
        subject_path: Path = self.subjects[idx]
        subject = subject_path.stem
        with open(subject_path / "registration_params.pickle", "rb") as f:
            params: dict = pickle.load(f)

        images = {}
        for view in self.views:
            nii_path = subject_path / f"{view}.nii.gz"
            if nii_path.is_file() and view in params:
                img = nib.load(str(nii_path))
                if len(img.shape) in (3, 4):
                    images[view] = img
        if not images or any(view not in images for view in self.target_views):
            return None
        candidates = [view for view in images if view not in self.target_views]
        if not candidates:
            return None
        input_views = set(candidates)
        if self.train and self.view_drop_probability > 0 and len(candidates) > 1:
            input_views = {view for view in candidates if random.random() >= self.view_drop_probability}
            if not input_views:
                input_views = {random.choice(candidates)}

        ok, frame = self._choose_frame(subject, images)
        if not ok:
            return None

        mm_to_su_affine = torch.as_tensor(params["mm_to_su_affine"])
        observation = {"mm_to_su_affine": mm_to_su_affine}
        crops = {}
        starts = {}
        tracker = MeanStdTracker()
        for view, img in images.items():
            data = load_frame(img, frame)  # [X, Y, Z]
            correction_file = subject_path / f"{view}_affine_correction.npy"
            if correction_file.is_file():
                correction = np.load(correction_file)
                self._check_correction(subject, view, params[view], correction)
                if correction.shape != (4, 4):
                    # Per-slice in-plane translations of a short-axis stack
                    # move the data itself, so they are applied here.
                    if correction.ndim != 2 or correction.shape[1] != 2:
                        raise ValueError(
                            f"{subject}/{view}: unexpected breathhold correction shape {correction.shape}"
                        )
                    data = translate_slices(data, correction)
                # A rigid long-axis correction changes only the affine, which
                # reaches the forward model through registration_params.pickle.
            observation[view] = dict(params[view])
            observation[view].setdefault("size_vox", torch.as_tensor(img.shape[:3]))

            y = torch.from_numpy(data)[None]  # [C, X, Y, Z]
            cropped, inside, start = crop_y(
                y,
                mm_to_su_affine.to(torch.float32),
                torch.as_tensor(params[view]["vox_to_mm_affine"]).to(torch.float32),
                self.geometry,
            )
            if 0 in cropped.shape:
                continue
            crops[view] = cropped
            starts[view] = start
            if view in input_views and (self.norm_views is None or view in self.norm_views):
                tracker.update_stats(cropped, mask=inside[None].expand_as(cropped))

        if not crops or tracker.N < 2:
            return None
        norm_mean, norm_std = tracker.get_final_stats()
        if not norm_std > 0:
            return None

        item_views = {}
        for view in self.views:
            if view not in crops:
                continue
            op = build_view_operator(observation, view, self.geometry, lanczos_a=self.lanczos_a)
            if op is None:
                continue
            y = crops[view]
            if tuple(op["row_mask"].shape) != tuple(y.shape[1:]):
                print(
                    f"{subject}/{view}: the forward model predicts {tuple(op['row_mask'].shape)} "
                    f"y-voxels but the crop is {tuple(y.shape[1:])}; leaving the view out"
                )
                continue
            noise_std_file = subject_path / f"{view}_noise_std.pt"
            if noise_std_file.exists():
                noise_std = float(torch.as_tensor(torch.load(noise_std_file, map_location="cpu")))
            else:
                noise_std = 0.0
            extra_noise_std = float(self.extra_noise_std.get(view, 0.0))
            y = (y - norm_mean) / norm_std * NORMALISED_STD
            item_views[view] = {
                "y": y[None].to(torch.float32).contiguous(),  # [1, C, X', Y', Z']
                "sigma_meas": noise_std / norm_std * NORMALISED_STD,
                "sigma_extra": extra_noise_std / norm_std * NORMALISED_STD,
                "op": op,
                "input": view in input_views,
                "crop_start": starts[view],
                "image_shape": tuple(int(s) for s in images[view].shape[:3]),
                "image_affine": torch.as_tensor(images[view].affine),
                # The whole acquired grid, not the crop: e.g. where a short-axis
                # stack reaches (`evaluate.stack_region`).
                "geometry": view_geometry(observation[view]),
            }
        if not any(d["input"] for d in item_views.values()):
            return None
        if any(view not in item_views for view in self.target_views):
            # A target the forward model cannot predict: nothing to train on.
            return None
        return {
            "subject": subject,
            "frame": frame,
            "mm_to_su_affine": mm_to_su_affine,
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "views": item_views,
        }


def collate(batch: list) -> list[dict]:
    r"""A batch is a list of items; subjects with no usable view are dropped."""
    return [item for item in batch if item is not None]
