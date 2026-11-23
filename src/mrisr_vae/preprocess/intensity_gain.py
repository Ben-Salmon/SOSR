r"""Intensity gain of each long-axis view relative to the short-axis slices.

    uv run mrisr-intensity-gain --data_dir <dir> [--line_mm 120] [--min_points 200]

The forward model slices one latent volume into every view, so it assumes the
views share an intensity scale. MR intensities have no absolute units, and
each series is scaled on its own (receive-coil selection and normalisation,
scanner adjustments): in the Kaggle CMR data the long-axis views are about
10-15% darker than the short-axis slices at the same anatomy, in every tissue
alike, while the long-axis views agree with each other.

For each long-axis view this estimates the gain g that makes it agree with the
short-axis slices where they intersect,

    g = mean(short axis) / mean(long axis)

over the points of its intersections with every short-axis slice, with the
breathhold correction on disk applied. A ratio of means, unlike a
least-squares slope, is not biased towards zero by leftover misalignment.

Writes <subject>/<view>_intensity_gain.json; `CMRObservationDataset`
multiplies the view, and its noise standard deviations, by g. Short-axis
slices are the reference and get no file. Run after mrisr-breathhold, and
again whenever it is rerun.
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from .breathhold.files import find_subjects
from .breathhold.optimise import LINE_MM, sample_intersections
from .breathhold.subject import NoValidLongAxisPlanes, SubjectData

MIN_POINTS = 200  # Fewer intersection points than this, and a view gets no gain.
# Gains outside this range are written, but reported as suspicious.
PLAUSIBLE = (0.5, 2.0)


def gain_path(subject: Path, view: str) -> Path:
    return Path(subject) / f"{view}_intensity_gain.json"


def load_intensity_gain(subject: Path, view: str) -> float:
    r"""The gain `mrisr-intensity-gain` saved for `view`, or 1 if none."""
    path = gain_path(subject, view)
    if not path.is_file():
        return 1.0
    with open(path) as f:
        return float(json.load(f)["gain"])


def estimate_gains(
    subject_data: SubjectData, voxel_transformations: torch.Tensor, line_mm: float | None = LINE_MM
) -> dict[str, dict]:
    r"""Per long-axis view: its gain relative to the short-axis slices
    ("gain"), and how many intersection points ("n_points") and short-axis
    slices ("n_slices") it rests on. `voxel_transformations`: the breathhold
    correction, (n_planes, 4, 4)."""
    names = subject_data.names
    pairs = subject_data.idx_pairs.to(subject_data.affines.device)
    # Pairs are (long axis, short axis) or (long axis, long axis).
    pairs = pairs[torch.tensor(["sa" in names[b] for b in pairs[:, 1].tolist()], device=pairs.device)]
    with torch.no_grad():
        long_axis, short_axis, _, _, mask, _ = sample_intersections(
            subject_data.images_pad,
            subject_data.segs_pad,
            subject_data.has_seg,
            subject_data.affines @ voxel_transformations,
            pairs,
            subject_data.shapes,
            subject_data.spacings,
            line_mm,
        )
    gains = {}
    for view in dict.fromkeys(names[a] for a in pairs[:, 0].tolist()):
        rows = torch.tensor([names[a] == view for a in pairs[:, 0].tolist()], device=pairs.device)
        m = mask[rows]
        long_sum = (long_axis[rows] * m).sum().item()
        short_sum = (short_axis[rows] * m).sum().item()
        gains[view] = {
            "gain": short_sum / long_sum if long_sum > 0 else float("nan"),
            "n_points": int(m.sum().item()),
            "n_slices": int(m.any(dim=2).any(dim=1).sum().item()),
        }
    return gains


def process_subject(subject_data: SubjectData, line_mm: float | None = LINE_MM, min_points: int = MIN_POINTS):
    r"""Writes each long-axis view's gain; returns the estimates, including
    views left without a file (too few points)."""
    transforms, uncorrected = subject_data.load_voxel_transformations()
    name = subject_data.subject_path.name
    if uncorrected:
        tqdm.write(f"{name}: no breathhold correction for {', '.join(uncorrected)}; using the acquired geometry")
    gains = estimate_gains(subject_data, transforms, line_mm)
    for view, estimate in gains.items():
        path = gain_path(subject_data.subject_path, view)
        if estimate["n_points"] < min_points or not estimate["gain"] > 0:
            tqdm.write(f"{name}/{view}: only {estimate['n_points']} intersection points; no gain written")
            path.unlink(missing_ok=True)  # Not a stale one from an earlier run either.
            continue
        if not PLAUSIBLE[0] <= estimate["gain"] <= PLAUSIBLE[1]:
            tqdm.write(f"{name}/{view}: gain {estimate['gain']:.3f} is outside {PLAUSIBLE}; check this subject")
        record = {
            **estimate,
            "reference": "short-axis slices",
            "line_mm": line_mm,
            # Views whose breathhold correction was missing when the gain was estimated.
            "uncorrected_views": uncorrected,
        }
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
    return gains


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="Directory of subject directories")
    parser.add_argument(
        "--line_mm",
        type=float,
        default=LINE_MM,
        help="Length of each intersection used, centred between the two images (0: half the largest image)",
    )
    parser.add_argument("--min_points", type=int, default=MIN_POINTS, help="Fewest intersection points for a gain")
    parser.add_argument("--subjects", nargs="+", help="Only these subject names")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    subjects = find_subjects(args.data_dir)
    if args.subjects is not None:
        subjects = [s for s in subjects if s.path.name in set(args.subjects)]
    all_gains = []
    for subject in tqdm(subjects, ncols=80):
        try:
            subject_data = SubjectData(subject, device=args.device)
        except NoValidLongAxisPlanes as error:
            tqdm.write(f"{error}, skipping")
            continue
        gains = process_subject(subject_data, args.line_mm or None, args.min_points)
        all_gains += [g["gain"] for g in gains.values() if g["n_points"] >= args.min_points]
    if all_gains:
        gains = torch.tensor(all_gains)
        print(
            f"{len(all_gains)} gains over {len(subjects)} subjects: median {gains.median():.3f}, "
            f"10th-90th percentile {gains.quantile(0.1):.3f}-{gains.quantile(0.9):.3f}"
        )


if __name__ == "__main__":
    main()
