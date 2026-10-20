r"""The dataset, on synthetic subject directories. CPU only."""

import pickle
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import random_split

from mrisr_vae.data import CMRObservationDataset, collate, split_subjects, translate_slices
from mrisr_vae.model import MRIVAE

from synthetic import geometry, write_subject

EXTRA = {"sa_1": 4.0, "la_2ch": 20.0}
DIFFUSION_PRIORS = Path(__file__).resolve().parents[2] / "diffusion-priors"


def test_item_matches_its_operators_and_feeds_the_model(tmp_path):
    write_subject(tmp_path)
    geom = geometry()
    dataset = CMRObservationDataset(tmp_path, geom, extra_noise_std=EXTRA)
    assert len(dataset) == 1
    item = dataset[0]
    assert list(item["views"]) == ["sa_1", "la_2ch"]
    assert 0 <= item["frame"] < 3  # a frame both views have
    assert dataset[0]["frame"] == item["frame"]  # fixed per subject outside training

    # The data are N(100, 20^2), normalised jointly to std 0.8.
    assert item["norm_mean"] == pytest.approx(100, abs=3)
    assert item["norm_std"] == pytest.approx(20, rel=0.1)
    values = torch.cat([d["y"].flatten() for d in item["views"].values()])
    assert abs(float(values.mean())) < 0.1
    assert float(values.std()) == pytest.approx(0.8, rel=0.1)

    for view, d in item["views"].items():
        assert d["y"].shape == (1, 1, *d["op"]["row_mask"].shape), view
        assert d["sigma_extra"] == pytest.approx(EXTRA[view] / item["norm_std"] * 0.8)
    assert item["views"]["sa_1"]["sigma_meas"] == pytest.approx(2.0 / item["norm_std"] * 0.8)
    assert item["views"]["la_2ch"]["sigma_meas"] == 0.0

    torch.manual_seed(0)
    model = MRIVAE(geom, n_layers=2, z_dim=4, n_filters=8, downsampling=[0, 1], checkpointed=False)
    out = model.elbo(collate([item]), amp=False)
    assert torch.isfinite(out["loss"])


def test_training_items_draw_frames_and_drop_views(tmp_path):
    write_subject(tmp_path)
    geom = geometry()
    random.seed(0)
    dataset = CMRObservationDataset(tmp_path, geom, train=True)
    frames = {dataset[0]["frame"] for _ in range(30)}
    assert len(frames) > 1 and frames <= {0, 1, 2}

    # Views are dropped from the encoder's input only, never all of them, and
    # only the input views set the normalisation.
    dropping = CMRObservationDataset(tmp_path, geom, train=True, view_drop_probability=1.0)
    for _ in range(10):
        item = dropping[0]
        assert len(item["views"]) == 2
        (shown,) = [view for view, d in item["views"].items() if d["input"]]
        alone = CMRObservationDataset(tmp_path, geom, views=[shown], frame=item["frame"])[0]
        assert item["norm_std"] == pytest.approx(alone["norm_std"])

    # Outside training nothing is dropped, whatever the probability.
    kept = CMRObservationDataset(tmp_path, geom, train=False, view_drop_probability=1.0)
    assert all(d["input"] for d in kept[0]["views"].values())


def test_view_selection_and_fixed_frame(tmp_path):
    write_subject(tmp_path)
    geom = geometry()
    item = CMRObservationDataset(tmp_path, geom, views=["sa_1"], frame=4)[0]
    assert list(item["views"]) == ["sa_1"] and item["frame"] == 4
    # la_2ch has only 3 frames
    assert CMRObservationDataset(tmp_path, geom, frame=4)[0] is None


def test_crop_start_places_y_in_the_image(tmp_path):
    subject = write_subject(tmp_path)
    item = CMRObservationDataset(tmp_path, geometry(), frame=1)[0]
    for view, d in item["views"].items():
        raw = nib.load(str(subject / f"{view}.nii.gz")).get_fdata()[..., 1]
        assert d["image_shape"] == raw.shape
        x0, y0, z0 = d["crop_start"]
        n0, n1, n2 = d["y"].shape[2:]
        expected = raw[x0 : x0 + n0, y0 : y0 + n1, z0 : z0 + n2]
        got = d["y"][0, 0].numpy() * item["norm_std"] / 0.8 + item["norm_mean"]
        assert np.allclose(got, expected, atol=1e-3), view


def test_normalisation_can_ignore_a_view(tmp_path):
    r"""Loading a held-out view with norm_views set leaves the other views as they were."""
    write_subject(tmp_path)
    geom = geometry()
    alone = CMRObservationDataset(tmp_path, geom, views=["sa_1"], frame=1)[0]
    together = CMRObservationDataset(tmp_path, geom, views=["sa_1", "la_2ch"], norm_views=["sa_1"], frame=1)[0]
    assert list(together["views"]) == ["sa_1", "la_2ch"]
    assert together["norm_mean"] == alone["norm_mean"] and together["norm_std"] == alone["norm_std"]
    assert torch.equal(together["views"]["sa_1"]["y"], alone["views"]["sa_1"]["y"])
    mixed = CMRObservationDataset(tmp_path, geom, views=["sa_1", "la_2ch"], frame=1)[0]
    assert mixed["norm_mean"] != alone["norm_mean"]


def test_views_outside_the_geometry_come_last(tmp_path):
    write_subject(tmp_path)
    sa_only = geometry(views=["sa_1"])
    dataset = CMRObservationDataset(tmp_path, sa_only, views=["la_2ch", "sa_1"])
    assert dataset.views == ["sa_1", "la_2ch"]
    assert list(dataset[0]["views"]) == ["sa_1", "la_2ch"]


def test_subject_outside_the_latent_is_dropped(tmp_path):
    subject = write_subject(tmp_path)
    with open(subject / "registration_params.pickle", "rb") as f:
        params = pickle.load(f)
    params["mm_to_su_affine"][:3, 3] = 1000.0
    with open(subject / "registration_params.pickle", "wb") as f:
        pickle.dump(params, f)
    dataset = CMRObservationDataset(tmp_path, geometry())
    assert dataset[0] is None
    assert collate([dataset[0]]) == []


def test_stale_breathhold_correction_raises(tmp_path):
    subject = write_subject(tmp_path, sa_correction=np.zeros((9, 2)))
    geom = geometry()
    assert CMRObservationDataset(tmp_path, geom)[0] is not None
    np.save(subject / "sa_1_affine_correction.npy", np.ones((9, 2)))
    with pytest.raises(RuntimeError):
        CMRObservationDataset(tmp_path, geom)[0]


def test_translate_slices():
    frame = np.arange(4 * 5 * 2, dtype=np.float32).reshape(4, 5, 2)
    correction = np.array([[1, 0], [0, -2]])
    out = translate_slices(frame, correction)
    assert np.array_equal(out[1:, :, 0], frame[:-1, :, 0]) and np.all(out[0, :, 0] == 0)
    assert np.array_equal(out[:, :-2, 1], frame[:, 2:, 1]) and np.all(out[:, -2:, 1] == 0)


def test_split_is_deterministic_and_matches_random_split(tmp_path):
    for i in range(20):
        write_subject(tmp_path, name=f"{i:03d}", views=("sa_1",), frames=(1,))
    train, val = split_subjects(tmp_path, ["sa_1", "la_2ch"], seed=42)
    assert (len(train), len(val)) == (18, 2)
    assert set(train).isdisjoint(val) and set(train) | set(val) == {f"{i:03d}" for i in range(20)}
    assert split_subjects(tmp_path, ["sa_1", "la_2ch"], seed=42) == (train, val)

    names = sorted(f"{i:03d}" for i in range(20))
    a, b = random_split(range(20), [0.9, 0.1], generator=torch.Generator().manual_seed(42))
    assert train == [names[i] for i in a.indices] and val == [names[i] for i in b.indices]


def test_split_matches_diffusion_priors(tmp_path):
    r"""The same validation subjects as diffusion-priors' build_datasets."""
    if not (DIFFUSION_PRIORS / "utils.py").exists():
        pytest.skip(f"{DIFFUSION_PRIORS} not found")
    for i in range(20):
        write_subject(tmp_path, name=f"{i:03d}", views=("sa_1",), frames=(1,))
    # build_datasets imports diffusion-priors' data.py when called, so the path
    # stays until then. That module needs torchvision, which this project does
    # not, so the test is skipped where it is missing.
    sys.path.insert(0, str(DIFFUSION_PRIORS))
    try:
        try:
            import forward_model

            forward_model.init_geometry(
                {"VIEWS": ["sa_1", "la_2ch"], "SCALE_FACTOR": 1.0, "RAW_SU_PER_LVOX": 2.0,
                 "x_size_su": [40, 40, 40], "X_MIN_COORDS_SU": [-20, -20, -20]}
            )
            from utils import build_datasets

            _, predictset, valset = build_datasets({"data_dir": str(tmp_path), "extra_noise_std": None})
        except ImportError as error:
            pytest.skip(f"cannot import diffusion-priors' data loading: {error}")
    finally:
        sys.path.remove(str(DIFFUSION_PRIORS))
    want_train = [predictset.dataset.subjects[i].stem for i in predictset.indices]
    want_val = [valset.dataset.subjects[i].stem for i in valset.indices]
    assert split_subjects(tmp_path, ["sa_1", "la_2ch"], seed=42) == (want_train, want_val)
