r"""Baselines scored like the VAE on a held-out view, from each subject's
lowest-numbered short-axis stack alone: interpolation of it (linear, cubic)
and SMORE (Remedios et al. 2023) on it, run unmodified from its pip-installed
package (slurm/smore.sh). Score the models on the same stack:
`mrisr-evaluate --conditioning_views sa_first --region sa_first`.

    uv run mrisr-baselines export  --checkpoint <ckpt> --data_dir <test> --out_dir <dir>
    (SMORE on each <dir>/<subject>/<stack>.nii.gz: slurm/smore.sh)
    uv run mrisr-baselines project --checkpoint <ckpt> --data_dir <test> --out_dir <dir> \
        --evaluation_dir <mrisr-evaluate --save_predictions --conditioning_views sa_first --region sa_first out_dir>
    uv run mrisr-baselines check   --checkpoint <ckpt> --data_dir <test> --out_dir <dir> --max_subjects 2

Every subcommand builds each subject's item as `mrisr-evaluate` does for the
`sa_first` conditioning set, with the latent geometry of the evaluated
checkpoint, so the subjects, the frame, the stack, its breathhold correction
and the normalisation are the evaluation's.

export   writes, per subject, the evaluated frame of its lowest-numbered
         short-axis stack, with its per-slice breathhold translations applied,
         as a 3D NIfTI on the stack's own grid (<out_dir>/<subject>/<stack>.nii.gz),
         a manifest.json, and <out_dir>/stacks.txt listing every stack for the
         SLURM array.

project  resamples each source onto the latent grid and pushes it through the
         held-out view's forward model, exactly as `mrisr-evaluate` projects
         the VAE's volume, writing <view>_pred_<source>.nii.gz beside the saved
         target and mask (<evaluation_dir>/sa_first/<subject>/) for `mrisr-score`.
         Sources: linear and cubic (trilinear and Keys tricubic interpolation
         of the exported stack) and smore (SMORE's output,
         <out_dir>/<subject>/smore/<stack>/<stack>_smore4.nii.gz, mapped to the
         stack's voxels by SMORE's own output affine). Latent voxels outside
         the stack take its edge value.

check    pushes each source's latent volume back through the stack's own
         forward model and scores it against the stack, with SMORE's output
         also shifted by +-0.5 input slices: the unshifted SMORE volume should
         agree best, or its grid is misplaced.
"""

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from .data import NORMALISED_STD, CMRObservationDataset, load_frame, translate_slices
from .evaluate import (
    heldout_projection,
    item_conditioning,
    lvox_to_stack_vox,
    norm_views_for,
    on_image_grid,
    resolve_conditioning,
    save_on_image_grid,
)
from .forward_model import BatchOperator, apply_affine
from .geometry import LatentGeometry
from .metrics import masked_metrics

SOURCES = ("linear", "cubic", "smore")
# Each subject's lowest-numbered short-axis stack; `mrisr-evaluate` writes this
# conditioning set's files under the same name.
CONDITIONING = "sa_first"
SMORE_SUFFIX = "_smore4"


# ----------------------------------------------------------------------------
# Interpolation at arbitrary points


def _linear_weights(t: torch.Tensor) -> list[torch.Tensor]:
    return [1 - t, t]


def _keys_weights(t: torch.Tensor, a: float = -0.5) -> list[torch.Tensor]:
    r"""Keys' cubic convolution weights for the taps at -1, 0, 1, 2 from
    floor(x), with t = x - floor(x). They sum to one and reproduce
    polynomials up to degree two."""
    def near(s):  # |s| <= 1
        return (a + 2) * s**3 - (a + 3) * s**2 + 1

    def far(s):  # 1 < |s| < 2
        return a * s**3 - 5 * a * s**2 + 8 * a * s - 4 * a

    return [far(1 + t), near(t), near(1 - t), far(2 - t)]


def sample(volume: torch.Tensor, coords: torch.Tensor, order: int) -> torch.Tensor:
    r"""`volume` (X, Y, Z) at points `coords` (N, 3), in voxels: trilinear
    (order 1) or Keys tricubic (order 3) interpolation. Taps outside the
    volume take the nearest edge voxel, so points outside it get the edge
    value."""
    if order == 1:
        offsets, weights = (0, 1), _linear_weights
    elif order == 3:
        offsets, weights = (-1, 0, 1, 2), _keys_weights
    else:
        raise ValueError(f"order must be 1 or 3, not {order}")
    volume = volume.to(torch.float64)
    coords = coords.to(device=volume.device, dtype=torch.float64)
    base = torch.floor(coords)
    t = coords - base
    base = base.long()
    sizes = volume.shape
    # Per axis, the clamped index and weight of each tap.
    taps = []
    for d in range(3):
        w = weights(t[:, d])
        idx = [torch.clamp(base[:, d] + o, 0, sizes[d] - 1) for o in offsets]
        taps.append(list(zip(idx, w)))
    flat = volume.reshape(-1)
    out = torch.zeros(len(coords), dtype=torch.float64, device=volume.device)
    for i0, w0 in taps[0]:
        for i1, w1 in taps[1]:
            w01 = w0 * w1
            row = (i0 * sizes[1] + i1) * sizes[2]
            for i2, w2 in taps[2]:
                out += w01 * w2 * flat[row + i2]
    return out


def latent_coords(geometry: LatentGeometry) -> torch.Tensor:
    r"""Every latent voxel's index (D * H * W, 3), in C order."""
    axes = [torch.arange(s, dtype=torch.float64) for s in geometry.size_lvox]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)


# ----------------------------------------------------------------------------
# Stacks and sources


def corrected_frame(subject_dir: Path, stack: str, frame: int | None) -> tuple[np.ndarray, nib.Nifti1Image]:
    r"""One frame of a short-axis stack with its per-slice breathhold
    translations applied, as `CMRObservationDataset.__getitem__` loads it."""
    img = nib.load(str(subject_dir / f"{stack}.nii.gz"))
    data = load_frame(img, frame)
    correction_file = subject_dir / f"{stack}_affine_correction.npy"
    if correction_file.is_file():
        correction = np.load(correction_file)
        if correction.shape != (4, 4):
            # The dataset has already checked its shape.
            data = translate_slices(data, correction)
    return data, img


def smore_path(out_dir: Path, subject: str, stack: str) -> Path:
    r"""Where `run-smore --out-dir <out_dir>/<subject>/smore` writes its output."""
    return Path(out_dir) / subject / "smore" / stack / f"{stack}{SMORE_SUFFIX}.nii.gz"


def source_volume(
    source: str, out_dir: Path, subject: str, stack: str
) -> tuple[torch.Tensor, torch.Tensor] | None:
    r"""A source's volume of `stack` and the affine (float64) from its voxels
    to the exported stack's voxels; None if SMORE's output is missing.

    SMORE's output grid relates to its input's through the two NIfTI affines
    SMORE itself writes (its own resampling convention), so no grid offset is
    derived here."""
    exported = nib.load(str(Path(out_dir) / subject / f"{stack}.nii.gz"))
    if source in ("linear", "cubic"):
        volume, to_input = np.asanyarray(exported.dataobj), np.eye(4)
    elif source == "smore":
        path = smore_path(out_dir, subject, stack)
        if not path.is_file():
            return None
        output = nib.load(str(path))
        volume = np.asanyarray(output.dataobj)
        to_input = np.linalg.inv(exported.affine) @ output.affine
    else:
        raise ValueError(f"Unknown source {source!r}")
    return torch.from_numpy(np.asarray(volume, dtype=np.float32)), torch.from_numpy(to_input)


def latent_volume(
    item: dict,
    stack: str,
    source: str,
    out_dir: Path,
    geometry: LatentGeometry,
    shift_slices: float = 0.0,
    device="cpu",
) -> torch.Tensor | None:
    r"""The source's volume of `stack`, raw intensities, on the latent grid
    (D, H, W), the stack's edge value outside it; None if SMORE's output is
    missing. `shift_slices` moves the source along the stack's slice axis, in
    input slices (`check`)."""
    loaded = source_volume(source, out_dir, item["subject"], stack)
    if loaded is None:
        return None
    volume, to_input = loaded
    order = 1 if source == "linear" else 3
    lvox_to_vox = lvox_to_stack_vox(item, stack, geometry)
    vox = apply_affine(lvox_to_vox, latent_coords(geometry).T).T  # (N, 3) in the stack's voxels
    shifted = vox - torch.tensor([0.0, 0.0, shift_slices], dtype=torch.float64)
    source_vox = apply_affine(torch.linalg.inv(to_input), shifted.T).T
    return sample(volume.to(device), source_vox, order).cpu().reshape(geometry.size_lvox)


def normalise(volume: torch.Tensor, item: dict) -> torch.Tensor:
    r"""Raw intensities -> the model's normalised units, (1, 1, D, H, W)."""
    return ((volume - item["norm_mean"]) / item["norm_std"] * NORMALISED_STD)[None, None].to(torch.float32)


# ----------------------------------------------------------------------------
# Items as mrisr-evaluate builds them


def load_geometry(checkpoint: str) -> tuple[LatentGeometry, int]:
    r"""The latent geometry and Lanczos order of an evaluated checkpoint."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    geometry = LatentGeometry.from_config(state["geometry_config"])
    lanczos_a = state.get("config", {}).get("model", {}).get("lanczos_a", 3)
    return geometry, lanczos_a


def evaluation_dataset(args, geometry: LatentGeometry) -> CMRObservationDataset:
    r"""The dataset `mrisr-evaluate` builds for the `sa_first` conditioning set
    (on a separate test set: every subject of it)."""
    candidates = resolve_conditioning(CONDITIONING, geometry.views)
    dataset = CMRObservationDataset(
        args.data_dir,
        geometry,
        subjects=args.subjects,
        views=list(geometry.views) + [args.view],
        norm_views=norm_views_for(CONDITIONING, candidates),
        train=False,
        frame=args.frame,
    )
    if args.max_subjects is not None:
        dataset.subjects = dataset.subjects[: args.max_subjects]
    return dataset


def conditioning_stack(item: dict) -> str | None:
    r"""The stack `mrisr-evaluate` conditions `item` on for `sa_first`: its
    lowest-numbered short-axis stack, which set its normalisation. None if
    that stack was left out of the item (e.g. the forward model's grid did
    not match its crop)."""
    stacks = item_conditioning(CONDITIONING, item, [])
    if len(stacks) != 1 or stacks[0] not in item["views"]:
        return None
    return stacks[0]


def items(dataset):
    for idx in tqdm(range(len(dataset)), ncols=80):
        item = dataset[idx]
        if item is not None:
            yield item


def read_manifest(out_dir: Path, subject: str) -> dict | None:
    path = Path(out_dir) / subject / "manifest.json"
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Subcommands


def stack_problems(img: nib.Nifti1Image, stack_geometry: dict) -> list[str]:
    r"""What would make SMORE misread an exported stack."""
    zooms = [float(z) for z in img.header.get_zooms()[:3]]
    problems = []
    if int(np.argmax(zooms)) != 2 or not zooms[2] > max(zooms[:2]):
        problems.append(f"the slice axis is not the coarsest: voxel sizes {zooms}")
    if not np.isclose(zooms[0], zooms[1], atol=1e-2):
        problems.append(f"in-plane voxels are not square: {zooms[:2]}")
    spacing = abs(float(stack_geometry["mm_per_vox"].flatten()[-1]))
    if not np.isclose(zooms[2], spacing, rtol=1e-3):
        problems.append(f"header slice spacing {zooms[2]} differs from the registered {spacing}")
    return problems


def export(args, geometry: LatentGeometry):
    dataset = evaluation_dataset(args, geometry)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stack_files, skipped = [], {}
    for item in items(dataset):
        subject = item["subject"]
        stack = conditioning_stack(item)
        if stack is None:
            skipped[subject] = [f"its lowest-numbered short-axis stack {item['norm_views']} is not usable"]
            continue
        data, img = corrected_frame(Path(args.data_dir) / subject, stack, item["frame"])
        nifti = nib.Nifti1Image(data.astype(np.float32), affine=img.affine)
        problems = stack_problems(nifti, item["views"][stack]["geometry"])
        if problems:
            skipped[f"{subject}/{stack}"] = problems
            continue
        path = out_dir / subject / f"{stack}.nii.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nifti, str(path))
        stack_files.append(str(path.resolve()))
        with open(out_dir / subject / "manifest.json", "w") as f:
            json.dump({"subject": subject, "frame": item["frame"], "stack": stack}, f, indent=2)
    with open(out_dir / "stacks.txt", "w") as f:
        f.writelines(f"{path}\n" for path in stack_files)
    print(f"Exported {len(stack_files)} stacks to {out_dir}; listed in {out_dir / 'stacks.txt'}")
    for name, problems in skipped.items():
        print(f"Skipped {name}: {'; '.join(problems)}")


def checked_manifest(args, item) -> str | None:
    r"""The exported stack of `item`, checked to be the evaluated frame and stack."""
    manifest = read_manifest(args.out_dir, item["subject"])
    if manifest is None:
        tqdm.write(f"{item['subject']}: not exported; skipped")
        return None
    if "stack" not in manifest:
        raise RuntimeError(f"{item['subject']}: exported by an older mrisr-baselines (every stack); rerun export")
    if manifest["frame"] != item["frame"]:
        raise RuntimeError(f"{item['subject']}: exported frame {manifest['frame']}, evaluated frame {item['frame']}")
    if manifest["stack"] != conditioning_stack(item):
        raise RuntimeError(
            f"{item['subject']}: exported {manifest['stack']}, but conditioned on {conditioning_stack(item)}"
        )
    return manifest["stack"]


def check_saved(path: Path, values: np.ndarray, what: str, atol: float):
    saved = nib.load(str(path)).get_fdata()
    if saved.shape != values.shape or not np.allclose(saved, values, atol=atol):
        raise RuntimeError(f"{path}: the recomputed {what} differs from the saved one")


def project(args, geometry: LatentGeometry, lanczos_a: int):
    dataset = evaluation_dataset(args, geometry)
    missing = {source: [] for source in args.sources}
    written = {source: 0 for source in args.sources}
    for item in items(dataset):
        subject = item["subject"]
        subject_dir = Path(args.evaluation_dir) / CONDITIONING / subject
        target_path = subject_dir / f"{args.view}_target.nii.gz"
        if not target_path.is_file():
            tqdm.write(f"{subject}: no saved target in {subject_dir}; skipped")
            continue
        stack = checked_manifest(args, item)
        if stack is None:
            continue
        projection = heldout_projection(
            item, args.view, geometry, lanczos_a=lanczos_a, region=args.region, device=args.device
        )
        if projection is None:
            continue
        heldout = item["views"][args.view]
        # The same voxels and target as the evaluation scored.
        check_saved(target_path, on_image_grid(projection.target, heldout), "target", atol=1e-2)
        check_saved(subject_dir / f"{args.view}_mask.nii.gz", on_image_grid(projection.mask, heldout), "mask", 1e-6)
        for source in args.sources:
            volume = latent_volume(item, stack, source, args.out_dir, geometry, device=args.device)
            if volume is None:
                missing[source].append(subject)
                continue
            with torch.no_grad():
                prediction = projection.project(normalise(volume, item).to(args.device))
            save_on_image_grid(str(subject_dir), item, args.view, prediction, f"pred_{source}")
            written[source] += 1
    for source in args.sources:
        print(f"{source}: wrote {written[source]} predictions")
        if missing[source]:
            print(f"{source}: no SMORE output for {len(missing[source])} subjects: {missing[source]}")


def check(args, geometry: LatentGeometry, lanczos_a: int):
    r"""Each source's agreement with the short-axis stack it came from."""
    dataset = evaluation_dataset(args, geometry)
    variants = [(source, 0.0) for source in args.sources]
    if "smore" in args.sources:
        variants += [("smore", -0.5), ("smore", 0.5)]
    rows = []
    for item in items(dataset):
        stack = checked_manifest(args, item)
        if stack is None:
            continue
        scale = item["norm_std"] / NORMALISED_STD
        A = BatchOperator([{stack: item["views"][stack]["op"]}], geometry.size_lvox, lanczos_a=lanczos_a)
        A = A.to(args.device)
        target = item["views"][stack]["y"].to(args.device)[0, 0] * scale + item["norm_mean"]
        mask = A.op(0, stack).row_mask
        for source, shift in variants:
            volume = latent_volume(item, stack, source, args.out_dir, geometry, shift_slices=shift, device=args.device)
            if volume is None:
                continue
            with torch.no_grad():
                predicted = A(normalise(volume, item).to(args.device))[0][0][0, 0] * scale + item["norm_mean"]
            metrics = masked_metrics(predicted * mask, target * mask, mask)
            rows.append({"subject": item["subject"], "stack": stack, "source": source, "shift": shift, **metrics})
    if not rows:
        print("Nothing to check")
        return
    out = Path(args.out_dir) / "check.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for source, shift in variants:
        selected = [r for r in rows if r["source"] == source and r["shift"] == shift]
        if selected:
            psnr = np.nanmean([r["psnr"] for r in selected])
            print(f"{source} shifted {shift:+.1f} slices: {len(selected)} stacks; mean psnr {psnr:.2f} against them")
    print(f"Wrote {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["export", "project", "check"])
    parser.add_argument("--checkpoint", required=True, help="The evaluated checkpoint: its latent geometry is used")
    parser.add_argument("--data_dir", required=True, help="The test set, as given to mrisr-evaluate")
    parser.add_argument("--out_dir", required=True, help="Exported stacks, manifests and SMORE outputs")
    parser.add_argument("--evaluation_dir", help="project: an mrisr-evaluate --save_predictions out_dir")
    parser.add_argument("--view", default="la_4ch", help="The held-out view. Default: la_4ch")
    parser.add_argument(
        "--region", default="sa_first", help="As mrisr-evaluate's --region; none: everywhere. Default: sa_first"
    )
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--subjects", nargs="+", help="Default: every subject of data_dir")
    parser.add_argument("--max_subjects", type=int)
    parser.add_argument("--frame", type=int, help="As mrisr-evaluate's --frame. Default: a frame fixed per subject")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.region in ("none", "null"):
        args.region = None
    if args.command == "project" and args.evaluation_dir is None:
        parser.error("project needs --evaluation_dir")

    geometry, lanczos_a = load_geometry(args.checkpoint)
    if args.command == "export":
        export(args, geometry)
    elif args.command == "project":
        project(args, geometry, lanczos_a)
    else:
        check(args, geometry, lanczos_a)


if __name__ == "__main__":
    main()
