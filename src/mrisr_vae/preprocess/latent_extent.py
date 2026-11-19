r"""How big to make the latent grid: the extent of the ROI over a dataset, in SU.

    uv run mrisr-latent-extent --dataset cmr --data_dir <train_dir> [--su_per_lvox 1.8] [--n_layers 8]

Run after `mrisr-register`. For every subject, maps its ROI (as
`mrisr-register` finds it: the heart's bounding box in the first CMR view,
after breathhold correction, or the bright voxels of a brain) into scanner
units with its registration parameters, and takes its lowest and highest
coordinate per axis. Prints the median voxel size of that view, the
`--percentiles` of the lowest and highest coordinates over subjects (low
percentile of the lows, high percentile of the highs), and the latent grid
they suggest at `--su_per_lvox`: its size in latent voxels, and the size the
Ladder VAE pads it to with `--n_layers` layers and `downsampling: auto`.

Copy the suggested `x_size_su` and `X_MIN_COORDS_SU` into the geometry yaml
(e.g. configs/kaggle_cmr_geometry.yaml), rounding as you see fit.
"""

import argparse
import math
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from ..model import auto_downsampling
from ..networks.utils import get_padded_size
from .registration import (
    BRAIN_VIEW,
    PICKLE,
    SkipSubject,
    apply_affine,
    brain_roi_coords_vox,
    cmr_roi_coords_vox,
    cmr_views,
    corrected_image,
)


def roi_extent(dataset: str, subject: Path):
    r"""(subject, (mm_per_vox, min_su, max_su) or None, reason)."""
    path = subject / PICKLE
    if not path.is_file():
        return subject.name, None, f"no {PICKLE}"
    with open(path, "rb") as f:
        params = pickle.load(f)
    try:
        if dataset == "cmr":
            views = cmr_views(subject)
            if not views:
                raise SkipSubject("no view")
            view = views[0]
            image, _ = corrected_image(subject, view, translate_data=True)
            coords_vox = cmr_roi_coords_vox(image)
        else:
            view = BRAIN_VIEW
            image_path = subject / f"{view}.nii.gz"
            if not image_path.is_file():
                raise SkipSubject(f"no {image_path.name}")
            coords_vox = brain_roi_coords_vox(nib.load(image_path))
    except SkipSubject as error:
        return subject.name, None, str(error)
    if view not in params:
        return subject.name, None, f"{view} is not in {PICKLE}: rerun mrisr-register"
    if coords_vox.shape[1] == 0:
        return subject.name, None, "empty ROI"
    coords_mm = apply_affine(params[view]["vox_to_mm_affine"].numpy(), coords_vox)
    coords_su = apply_affine(params["mm_to_su_affine"].numpy(), coords_mm)
    mm_per_vox = np.abs(params[view]["mm_per_vox"].numpy())
    return subject.name, (mm_per_vox, coords_su.min(axis=1), coords_su.max(axis=1)), ""


def suggest_grid(min_su: np.ndarray, max_su: np.ndarray, su_per_lvox: float, n_layers: int) -> dict:
    r"""The latent grid covering [min_su, max_su]: its yaml values, size in
    latent voxels, and the size the Ladder VAE pads it to."""
    x_size_su = [math.ceil(float(hi - lo)) for lo, hi in zip(min_su, max_su)]
    x_min_su = [math.floor(float(lo)) for lo in min_su]
    size_lvox = [round(s / su_per_lvox) for s in x_size_su]
    downsampling = auto_downsampling(size_lvox, n_layers)
    n_downsample = [sum(layer[i] for layer in downsampling) for i in range(3)]
    return {
        "x_size_su": x_size_su,
        "X_MIN_COORDS_SU": x_min_su,
        "size_lvox": size_lvox,
        "padded_lvox": get_padded_size(size_lvox, n_downsample),
        "halvings": n_downsample,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["cmr", "brain"])
    parser.add_argument("--data_dir", required=True, help="The training subjects")
    parser.add_argument("--su_per_lvox", type=float, default=1.8, help="The geometry's RAW_SU_PER_LVOX / SCALE_FACTOR")
    parser.add_argument("--n_layers", type=int, default=8, help="The Ladder VAE's model: n_layers")
    parser.add_argument("--percentiles", type=float, nargs=2, default=[10, 90], metavar=("LOW", "HIGH"))
    parser.add_argument("--n_jobs", type=int, default=16)
    args = parser.parse_args(argv)

    subjects = sorted(Path(args.data_dir).glob("[0-9]*"))
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(roi_extent)(args.dataset, subject) for subject in tqdm(subjects, ncols=80)
    )
    extents = [extent for _, extent, _ in results if extent is not None]
    for name, extent, reason in results:
        if extent is None:
            print(f"{name} skipped: {reason}")
    if not extents:
        raise SystemExit("No subject had an ROI")

    mm_per_vox = np.stack([e[0] for e in extents])
    lows = np.stack([e[1] for e in extents])
    highs = np.stack([e[2] for e in extents])
    low_p, high_p = args.percentiles
    min_su = np.percentile(lows, low_p, axis=0)
    max_su = np.percentile(highs, high_p, axis=0)
    grid = suggest_grid(min_su, max_su, args.su_per_lvox, args.n_layers)

    np.set_printoptions(precision=2, suppress=True)
    print(f"{len(extents)} of {len(subjects)} subjects")
    print(f"Median mm per voxel of the ROI view: {np.median(mm_per_vox, axis=0)}")
    print(f"ROI lowest coordinate, {low_p:g}th percentile (SU):  {min_su}")
    print(f"ROI highest coordinate, {high_p:g}th percentile (SU): {max_su}")
    print(f"ROI size (SU): {max_su - min_su}")
    print(
        f"At {args.su_per_lvox} SU per latent voxel: {grid['size_lvox']} latent voxels, padded to "
        f"{grid['padded_lvox']} for {grid['halvings']} halvings ({args.n_layers} layers, downsampling: auto)"
    )
    print("\nSuggested geometry yaml lines:")
    print(f"x_size_su: {grid['x_size_su']}")
    print(f"X_MIN_COORDS_SU: {grid['X_MIN_COORDS_SU']}")


if __name__ == "__main__":
    main()
