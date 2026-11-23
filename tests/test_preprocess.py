r"""Breathhold correction, registration and the latent extent, on synthetic
subjects. CPU only.

Where ../diffusion-priors is present, the refactored code is also checked
against the original scripts there (registration/):
the same pickles.
"""

import importlib
import pickle
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from mrisr_vae.data import CMRObservationDataset, translate_slices
from mrisr_vae.preprocess import intensity_gain, latent_extent, registration
from mrisr_vae.preprocess.breathhold import optimise
from mrisr_vae.preprocess.breathhold.files import find_subjects
from mrisr_vae.preprocess.breathhold.subject import SubjectData

from synthetic import GEOMETRY_CONFIG, VIEW_GEOMETRY, geometry, grid_affine, rotation

DIFFUSION_PRIORS = Path(__file__).resolve().parents[2] / "diffusion-priors"

# A short-axis stack with a wide field of view, so the heart ROI (8000 mm^2)
# is a box inside it, and two long-axis views through it.
CMR_GEOMETRY = {
    "sa_1": ((96, 96, 9), [1.5, 1.5, 6.0], None, [0.0, 0.0, 0.0]),
    "la_2ch": VIEW_GEOMETRY["la_2ch"],
    "la_4ch": VIEW_GEOMETRY["la_4ch"],
}
# In-plane, in sa_1's voxels: 64^2 pixels, more than the ROI's 8000 mm^2
# (3556 pixels), so the ROI's pixels are all inside it and its box is this box.
HEART = (slice(16, 80), slice(16, 80))


def write_cmr_subject(root: Path, name="1001", n_frames=6, seed=0, sa_shift=None, la_correction=None) -> Path:
    r"""4D views whose intensity varies over time inside the heart box only;
    optionally breathhold corrections."""
    subject = Path(root) / name
    subject.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for view, (shape, spacing, angles, centre) in CMR_GEOMETRY.items():
        axes = np.eye(3) if angles is None else rotation(*angles)
        # Constant over time outside the heart.
        data = np.repeat(100 + rng.standard_normal((*shape, 1)).astype(np.float32), n_frames, axis=-1)
        if view == "sa_1":
            data[HEART] += 50 * rng.standard_normal((*data[HEART].shape[:-1], n_frames))
        nib.save(nib.Nifti1Image(data, grid_affine(shape, spacing, axes, centre)), str(subject / f"{view}.nii.gz"))
    if sa_shift is not None:
        np.save(subject / "sa_1_affine_correction.npy", np.asarray(sa_shift, dtype=np.float32))
    if la_correction is not None:
        np.save(subject / "la_2ch_affine_correction.npy", np.asarray(la_correction, dtype=np.float32))
    return subject


def shifts():
    shift = np.zeros((9, 2), dtype=np.float32)
    shift[3] = [2, -1]
    shift[6] = [-3, 0]
    return shift


def small_rigid():
    correction = np.eye(4, dtype=np.float32)
    correction[:3, 3] = [0.5, -1.0, 0.0]
    return correction


@contextmanager
def old_modules(directory: Path, names):
    r"""Imports the original scripts from `directory` as top-level modules,
    and forgets them afterwards."""
    if not directory.is_dir():
        pytest.skip(f"{directory} is missing")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    sys.path.insert(0, str(directory))
    try:
        yield [importlib.import_module(name) for name in names]
    finally:
        sys.path.remove(str(directory))
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def read_pickle(subject: Path) -> dict:
    with open(subject / "registration_params.pickle", "rb") as f:
        return pickle.load(f)


def test_cmr_registration_gives_what_the_dataset_reads(tmp_path):
    subject = write_cmr_subject(tmp_path, sa_shift=shifts(), la_correction=small_rigid())
    assert registration.process_subject("cmr", subject, check=False) == (subject.name, "written", [])
    params = read_pickle(subject)
    assert list(params) == ["sa_1", "la_4ch", "la_2ch", "mm_to_su_affine"]
    for view in ("sa_1", "la_4ch", "la_2ch"):
        assert params[view]["vox_to_mm_affine"].dtype == torch.float64
        assert params[view]["size_vox"].tolist() == list(CMR_GEOMETRY[view][0])
    assert torch.equal(params["sa_1"]["affine_correction"], torch.from_numpy(shifts().astype(np.float64)))
    # The rigid correction moved la_2ch's affine; the per-slice shifts did not move sa_1's.
    raw = nib.load(str(subject / "la_2ch.nii.gz")).affine
    assert np.allclose(params["la_2ch"]["vox_to_mm_affine"].numpy(), raw @ small_rigid(), atol=1e-4)
    assert np.allclose(params["sa_1"]["vox_to_mm_affine"].numpy(), nib.load(str(subject / "sa_1.nii.gz")).affine, atol=1e-4)
    # The centre is the heart box's: the shifts move only two of the nine slices.
    box_centre_vox = [(16 + 79) / 2, (16 + 79) / 2, 4]
    centre_mm = nib.affines.apply_affine(nib.load(str(subject / "sa_1.nii.gz")).affine, box_centre_vox)
    assert np.allclose(-params["mm_to_su_affine"][:3, 3].numpy(), centre_mm, atol=1e-3)

    # The dataset loads it, checks the recorded corrections against the files, and builds every operator.
    geom = geometry(views=["sa_1", "la_2ch"])
    item = CMRObservationDataset(tmp_path, geom, views=["sa_1", "la_2ch", "la_4ch"], frame=0)[0]
    assert set(item["views"]) == {"sa_1", "la_2ch", "la_4ch"}

    # --check: nothing differs, then a change is found.
    assert registration.process_subject("cmr", subject, check=True)[1] == "same"
    np.save(subject / "sa_1_affine_correction.npy", np.zeros((9, 2), dtype=np.float32))
    name, status, details = registration.process_subject("cmr", subject, check=True)
    assert status == "differs" and any(d.startswith("sa_1/affine_correction") for d in details)


def test_a_subject_without_a_heart_roi_is_skipped(tmp_path):
    subject = write_cmr_subject(tmp_path)
    img = nib.load(str(subject / "sa_1.nii.gz"))
    nib.save(nib.Nifti1Image(img.get_fdata()[..., 0], img.affine), str(subject / "sa_1.nii.gz"))  # 3D: no cycle
    name, status, details = registration.process_subject("cmr", subject, check=False)
    assert status == "skipped" and "4-D" in details[0]
    assert not (subject / "registration_params.pickle").exists()


def test_translate_slices_matches_the_original_correction(tmp_path):
    with old_modules(DIFFUSION_PRIORS / "registration", ["reg_utils"]) as (reg_utils,):
        subject = write_cmr_subject(tmp_path)
        img = nib.load(str(subject / "sa_1.nii.gz"))
        old = reg_utils.apply_affine_correction(img, shifts()).get_fdata()
    assert np.array_equal(translate_slices(img.get_fdata(), shifts()), old)


def test_cmr_registration_matches_the_original(tmp_path):
    names = ["reg_utils", "extract_heart_bounding_box", "get_registration_params_kaggle_cmr"]
    with old_modules(DIFFUSION_PRIORS / "registration", names) as (_, _, old):
        for i, corrections in enumerate([{}, {"sa_shift": shifts(), "la_correction": small_rigid()}]):
            subject = write_cmr_subject(tmp_path / f"old_{i}", seed=i, **corrections)
            old.process_subject(subject)
            new = registration.register_cmr(subject)
            assert registration.differences(new, read_pickle(subject)) == []


def test_brain_registration_matches_the_original(tmp_path):
    subject = tmp_path / "2001"
    subject.mkdir()
    shape, spacing = (40, 44, 12), [0.7, 0.7, 5.0]
    data = np.zeros(shape, dtype=np.float32)
    data[10:30, 12:36, 2:10] = 200
    nib.save(nib.Nifti1Image(data, grid_affine(shape, spacing, rotation(5, -3, 10), [3, -2, 1])), str(subject / "ax.nii.gz"))
    new = registration.register_brain(subject)
    assert set(new) == {"ax", "mm_to_su_affine"}
    with old_modules(DIFFUSION_PRIORS / "registration", ["reg_utils", "get_registration_params_brain"]) as (_, old):
        old.process_subject(subject)
        assert registration.differences(new, read_pickle(subject)) == []


def write_breathhold_subject(root: Path, name="1001", seed=0) -> Path:
    r"""A short-axis stack and a long-axis slice through all of its slices, of
    smooth random intensity."""
    subject = Path(root) / name
    subject.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for view in ("sa_1", "la_4ch"):
        shape, spacing, angles, centre = VIEW_GEOMETRY[view]
        axes = np.eye(3) if angles is None else rotation(*angles)
        noise = torch.from_numpy(rng.standard_normal((1, 1, *shape)).astype(np.float32))
        smooth = torch.nn.functional.avg_pool3d(noise, (5, 5, 1), stride=1, padding=(2, 2, 0))[0, 0].numpy()
        data = (100 + 200 * smooth)[..., None].repeat(2, axis=-1)
        nib.save(nib.Nifti1Image(data, grid_affine(shape, spacing, axes, centre)), str(subject / f"{view}.nii.gz"))
    return subject


# A short-axis stack and two orthogonal long-axis planes through all of its
# slices, as in CMR, with larger fields of view than VIEW_GEOMETRY.
MISREGISTERED_GEOMETRY = {
    "sa_1": ((64, 64, 9), [1.5, 1.5, 6.0], None, [0.0, 0.0, 0.0]),
    "la_4ch": ((64, 64, 1), [1.5, 1.5, 6.0], (90, 0, 15), [0.0, 0.0, 0.0]),
    "la_2ch": ((64, 64, 1), [1.5, 1.5, 6.0], (90, 0, 105), [0.0, 0.0, 0.0]),
}


def write_misregistered_subject(root: Path, sa_shifts, segmented: bool, name="1001", seed=0) -> Path:
    r"""Views sampled from one volume -- random blobs and an ellipsoidal heart --
    with each short-axis slice's content displaced by `sa_shifts` (n_slices, 2)
    voxels, so that its correct breathhold correction is `sa_shifts`. With
    `segmented`, heart segmentations of sa_1 and la_4ch (none for la_2ch)."""
    subject = Path(root) / name
    subject.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-35, 35, (60, 3))
    widths = rng.uniform(3, 8, 60)
    amplitudes = rng.uniform(0.5, 1.5, 60)

    def volume(points):
        d2 = ((points[..., None, :] - centres) ** 2).sum(-1)
        # A soft edge, about 4 mm wide, like a partial-volumed one.
        radius = np.sqrt((((points - [3.0, -2.0, 0.0]) / [16.0, 13.0, 30.0]) ** 2).sum(-1))
        heart = 1 / (1 + np.exp((radius - 1) / 0.15))
        image = 100 * (amplitudes * np.exp(-d2 / (2 * widths**2))).sum(-1) + 150 * heart
        return image, (heart > 0.5).astype(np.float32)

    for view, (shape, spacing, angles, centre) in MISREGISTERED_GEOMETRY.items():
        axes = np.eye(3) if angles is None else rotation(*angles)
        affine = grid_affine(shape, spacing, axes, centre)
        i, j, k = np.meshgrid(*(np.arange(n, dtype=float) for n in shape), indexing="ij")
        if view.startswith("sa"):
            i = i + np.asarray(sa_shifts)[None, None, :, 0]
            j = j + np.asarray(sa_shifts)[None, None, :, 1]
        points = np.stack([i, j, k, np.ones_like(i)], axis=-1) @ affine.T
        image, heart = volume(points[..., :3])
        nib.save(nib.Nifti1Image(image[..., None].repeat(2, axis=-1), affine), str(subject / f"{view}.nii.gz"))
        if segmented and view == "sa_1":
            nib.save(nib.Nifti1Image(heart, affine), str(subject / "seg_sa_ED.nii.gz"))
        if segmented and view == "la_4ch":
            nib.save(nib.Nifti1Image(heart, affine), str(subject / "seg_la_4ch_ED.nii.gz"))
    return subject


@pytest.mark.parametrize("segmented", [True, False])
def test_breathhold_recovers_large_short_axis_shifts(tmp_path, segmented):
    # Up to 6 voxels (9 mm) per slice: beyond what gradient descent on the
    # translations reliably recovered.
    shifts = np.random.default_rng(1).integers(-6, 7, (9, 2))
    write_misregistered_subject(tmp_path, shifts, segmented)
    (files,) = find_subjects(tmp_path)
    data = SubjectData(files, device="cpu")
    assert data.has_seg.tolist() == [segmented, False] + [segmented] * 9  # la_4ch, la_2ch, sa_1 slices.
    result = optimise.optimize_affines(data, search_radius=8, lax_steps=50, save=False)
    sa = np.array(["sa" in n for n in data.names])
    recovered = result.voxel_transformations[sa, :2, 3].numpy()
    assert np.array_equal(recovered, shifts)
    assert result.best_loss < result.initial_loss


def test_intensity_gain_recovers_a_darker_long_axis(tmp_path):
    # Misregistered short-axis slices and long-axis views 20% darker: the gain
    # is estimated on the breathhold correction saved on disk.
    shifts = np.random.default_rng(2).integers(-4, 5, (9, 2))
    subject = write_misregistered_subject(tmp_path, shifts, segmented=False)
    for view in ("la_4ch", "la_2ch"):
        image = nib.load(subject / f"{view}.nii.gz")
        nib.save(nib.Nifti1Image(0.8 * image.get_fdata(), image.affine), str(subject / f"{view}.nii.gz"))
    optimise.main(["--data_dir", str(tmp_path), "--search_radius", "6", "--lax_steps", "20", "--device", "cpu"])
    intensity_gain.main(["--data_dir", str(tmp_path), "--device", "cpu"])
    for view in ("la_4ch", "la_2ch"):
        assert intensity_gain.load_intensity_gain(subject, view) == pytest.approx(1.25, rel=0.02), view
    assert not intensity_gain.gain_path(subject, "sa_1").exists()
    assert intensity_gain.load_intensity_gain(subject, "sa_1") == 1.0


def test_breathhold_cli_writes_corrections_the_registration_reads(tmp_path):
    subject = write_breathhold_subject(tmp_path)
    optimise.main(
        ["--data_dir", str(tmp_path), "--rounds", "1", "--search_radius", "2", "--lax_steps", "3", "--device", "cpu"]
    )
    sa = np.load(subject / "sa_1_affine_correction.npy")
    la = np.load(subject / "la_4ch_affine_correction.npy")
    assert sa.shape == (9, 2) and np.array_equal(sa, np.round(sa)) and la.shape == (4, 4)


def test_latent_extent_suggests_a_grid(tmp_path):
    subject = write_cmr_subject(tmp_path)
    registration.process_subject("cmr", subject, check=False)
    name, (mm_per_vox, low, high), reason = latent_extent.roi_extent("cmr", subject)
    assert np.allclose(mm_per_vox, [1.5, 1.5, 6.0])
    # The ROI is centred on the origin of SU, and spans the heart box in-plane.
    assert np.allclose(low + high, 0, atol=1.6)
    assert np.allclose(high - low, [63 * 1.5, 63 * 1.5, 8 * 6.0], atol=1e-3)
    grid = latent_extent.suggest_grid(low, high, su_per_lvox=2.0, n_layers=2)
    assert grid["x_size_su"] == [int(np.ceil(s)) for s in high - low]
    assert all(p >= s for p, s in zip(grid["padded_lvox"], grid["size_lvox"]))
    assert latent_extent.roi_extent("cmr", tmp_path / "missing")[1] is None


def test_geometry_config_matches_latent_extent_format():
    # The yaml keys mrisr-latent-extent prints are the ones LatentGeometry reads.
    grid = latent_extent.suggest_grid(np.array([-20.0] * 3), np.array([20.0] * 3), 2.0, 2)
    config = dict(GEOMETRY_CONFIG, x_size_su=grid["x_size_su"], X_MIN_COORDS_SU=grid["X_MIN_COORDS_SU"])
    assert list(geometry().size_lvox) == grid["size_lvox"]
    assert config == GEOMETRY_CONFIG


def test_subjects_without_long_axis_are_not_listed(tmp_path):
    write_breathhold_subject(tmp_path, name="1001")
    sa_only = write_breathhold_subject(tmp_path, name="1002")
    (sa_only / "la_4ch.nii.gz").unlink()
    assert [s.path.name for s in find_subjects(tmp_path)] == ["1001"]
    shutil.rmtree(tmp_path / "1001")
    with pytest.raises(RuntimeError):
        find_subjects(tmp_path)
