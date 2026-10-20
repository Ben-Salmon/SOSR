r"""Masked MSE, PSNR and SSIM. CPU only."""

import math

import numpy as np
import pytest

from mrisr_vae.metrics import masked_metrics, masked_spearman, masked_ssim


def smooth_image(shape=(40, 40, 1), seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(np.cumsum(rng.standard_normal(shape), axis=0), axis=1)
    return 100 + 10 * x / x.std()


def test_identical_images():
    target = smooth_image()
    mask = np.ones(target.shape)
    metrics = masked_metrics(target, target, mask)
    assert metrics["mse"] == 0 and math.isinf(metrics["psnr"])
    assert metrics["ssim"] == pytest.approx(1.0)


def test_psnr_from_mse_and_target_range():
    target = smooth_image()
    prediction = target + 2.0
    mask = np.zeros(target.shape)
    mask[5:35, 5:35] = 1
    metrics = masked_metrics(prediction, target, mask)
    masked = target[mask > 0]
    assert metrics["mse"] == pytest.approx(4.0)
    assert metrics["data_range"] == pytest.approx(masked.max() - masked.min())
    assert metrics["psnr"] == pytest.approx(10 * math.log10(metrics["data_range"] ** 2 / 4.0))


def test_ssim_falls_with_noise_and_ignores_voxels_off_the_mask():
    target = smooth_image()
    rng = np.random.default_rng(1)
    noisy = target + 5 * rng.standard_normal(target.shape)
    mask = np.zeros(target.shape)
    mask[5:35, 5:35] = 1
    data_range = float(np.ptp(target[mask > 0]))
    clean_ssim = masked_ssim(target, target, mask, data_range)
    noisy_ssim = masked_ssim(noisy, target, mask, data_range)
    assert clean_ssim == pytest.approx(1.0) and noisy_ssim < 0.99
    # Anything off the mask (a prediction is zero there) changes nothing.
    off_mask = noisy.copy()
    off_mask[mask == 0] = 0
    assert masked_ssim(off_mask, target, mask, data_range) == pytest.approx(noisy_ssim)


def test_ssim_is_undefined_when_no_window_fits_the_mask():
    target = smooth_image()
    mask = np.zeros(target.shape)
    mask[5:14, 5:35] = 1  # 9 rows: narrower than the 11-pixel window
    assert math.isnan(masked_ssim(target, target, mask, 1.0))


def test_slices_are_scored_separately():
    r"""A stack is scored in-plane: SSIM of two identical slices equals that of one."""
    one = smooth_image((40, 40, 1))
    rng = np.random.default_rng(2)
    noisy = one + 3 * rng.standard_normal(one.shape)
    mask = np.ones(one.shape)
    single = masked_ssim(noisy, one, mask, 50.0)
    stacked = masked_ssim(np.concatenate([noisy, noisy], 2), np.concatenate([one, one], 2), np.ones((40, 40, 2)), 50.0)
    assert stacked == pytest.approx(single)


def test_masked_spearman():
    a = smooth_image(seed=1)
    mask = np.zeros(a.shape)
    mask[5:35, 5:35] = 1
    # Any increasing function correlates perfectly, any decreasing one perfectly negatively.
    assert masked_spearman(a, np.exp(a / 10), mask) == pytest.approx(1.0)
    assert masked_spearman(a, -(a**3), mask) == pytest.approx(-1.0)
    # Voxels off the mask do not count.
    b = a.copy()
    b[mask == 0] = -b[mask == 0]
    assert masked_spearman(a, b, mask) == pytest.approx(1.0)
    assert math.isnan(masked_spearman(a, np.full(a.shape, 3.0), mask))
    assert math.isnan(masked_spearman(a, a, np.zeros(a.shape)))
