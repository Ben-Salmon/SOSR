r"""Scores a trained VAE on views it never saw: the config's `heldout_views`.

    uv run mrisr-evaluate --config configs/kaggle_cmr.yaml [options]
    uv run mrisr-evaluate --checkpoint <checkpoint_dir>/best.pt --out_dir <dir> \
        [--conditioning_views all sa la_2ch] [--split val] [--save_predictions]

With --config, the config's `evaluate` section (see `train.DEFAULTS`) sets the
checkpoint (best or last, in its checkpoint_dir), out_dir, data_dir, split,
conditioning_views, frame, n_samples, seed, save_predictions, direct and region; options
given on the command line override it. Posterior samples are drawn with the random
seed reset per subject and conditioning set, from `seed` and their names, so
a subject's scores do not depend on which other subjects are evaluated.

For each subject with a held-out view, and each set of conditioning views, the
model is given those views only -- normalised from those views alone, so
nothing else leaks into the input -- and its super-resolved volume is pushed
through the held-out view's forward model. The prediction is scored against
the held-out view, in raw intensities, on the voxels that forward model
supports (see `metrics` for mse, psnr and ssim).

With `region` (e.g. `sa`), only on those of them inside the subject's tallest
stack of that view group, of the views the model was trained on, the highest
index (the most recent) breaking ties: within its in-plane field of view, and
from half a slice thickness below its first slice to half a thickness above
its last, gaps between slices included. Elsewhere, e.g. the atria above a
short-axis stack, no training view has data, so a model trained without the
held-out view has nothing to go on there. The region depends on the stack's
geometry only, so it is the same for every model and conditioning set, even
those without the stack. Subjects without such a stack, or with no held-out
voxel inside it, are skipped and listed.

Conditioning sets: each argument of --conditioning_views is `all` (every view
the model was trained on), or a comma-separated list of views or view groups,
e.g. `sa` (every short-axis stack), `la_2ch`, or `sa,la_2ch`. Every set is
evaluated on the same frame of each subject.

Writes <out_dir>/metrics.csv, one row per subject, held-out view and
conditioning set, with `n_voxels` scored (and, with `region`, the
`region_stack` bounding them), and each metric for:

    sample       each posterior sample, averaged over the samples
    sample_mean  the mean of the samples (the MMSE estimate)
    direct       the direct predictor's estimate (if the model has one, and
                 `direct` is on: off, only the samples are scored)
    supervised   for the supervised baseline (`model: type: supervised`),
                 its estimate, in place of all of these

and, with 2 or more samples, the uncertainty of the sample mean in the view:
the per-voxel standard deviation of the samples pushed through the view's
forward model (exact for a linear model, unlike pushing the latent's std
through it), as `mean_std` over the scored voxels, and `spearman_std_error`,
its rank correlation with the absolute error of the sample mean there.

The supervised baseline is trained against its target view -- held out from
its input, not from its training -- so score it on a separate test set
(--data_dir), not on the training data's subjects.

With --save_predictions, also writes, in <out_dir>/<conditioning>/<subject>/,
on the held-out view's own image grid (its NIfTI affine and shape), in raw
intensities, for the evaluated frame (header toffset):

    <view>_target.nii.gz           the held-out view
    <view>_pred_sample_mean.nii.gz the prediction from the mean of the samples
    <view>_pred_direct.nii.gz      the direct predictor's prediction
    <view>_pred_supervised.nii.gz  the supervised baseline's prediction
    <view>_sample_std.nii.gz       the projected samples' per-voxel std (2+ samples)
    <view>_mask.nii.gz             1 on the scored voxels, 0 elsewhere

Put another method's predictions beside them as <view>_pred_<name>.nii.gz
and score them with `mrisr-score --name <name>`, which uses the same code.
"""

import argparse
import csv
import os
import zlib
from collections.abc import Callable
from typing import NamedTuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import NORMALISED_STD, CMRObservationDataset, collate, split_subjects
from .forward_model import BatchOperator, apply_affine, invert_affine
from .geometry import LatentGeometry
from .likelihood import view_group
from .metrics import masked_metrics, masked_spearman
from .model import MRIVAE, build_model
from .train import options_from_config


def resolve_conditioning(spec: str, views: list[str]) -> list[str]:
    r"""The views of a conditioning set: `all`, or comma-separated views or groups."""
    if spec == "all":
        return list(views)
    names = [name for name in spec.split(",") if name]
    for name in names:
        if not any(view == name or view_group(view) == name for view in views):
            raise ValueError(f"`{name}` matches none of the views the model was trained on: {views}")
    return [view for view in views if any(view == name or view_group(view) == name for name in names)]


def conditioning_label(spec: str) -> str:
    return spec.replace(",", "+")


def stack_height_mm(geometry: dict) -> float:
    r"""Through-plane extent of a stack: from half a slice thickness below its
    first slice centre to half a thickness above its last."""
    n_slices = int(geometry["size_vox"][2])
    spacing = abs(float(geometry["mm_per_vox"].flatten()[-1]))
    thickness = float(geometry["kernel_size_mm"].flatten()[-1])
    return (n_slices - 1) * spacing + thickness


def _view_index(view: str):
    suffix = view.rsplit("_", 1)[-1]
    return (int(suffix) if suffix.isdigit() else -1, view)


def region_stack(item: dict, group: str, views) -> str | None:
    r"""The tallest stack of `group` (e.g. `sa`) among the item's `views`, the
    highest index (the most recent) breaking ties; None if it has none."""
    candidates = [view for view in item["views"] if view in views and view_group(view) == group]
    if not candidates:
        return None
    heights = {view: stack_height_mm(item["views"][view]["geometry"]) for view in candidates}
    tallest = max(heights.values())
    return max((view for view in candidates if heights[view] >= tallest - 1e-3), key=_view_index)


def lvox_to_stack_vox(item: dict, stack: str, geometry: LatentGeometry) -> torch.Tensor:
    r"""The affine (float64) from latent voxels to `stack`'s voxels, through the
    stack's registered geometry, as the forward model sees it."""
    g = item["views"][stack]["geometry"]
    mm_to_su = torch.as_tensor(item["mm_to_su_affine"]).to(torch.float64)
    vox_to_su = mm_to_su @ g["vox_to_mm_affine"]
    return invert_affine(vox_to_su) @ geometry.lvox_to_su_affine.to(torch.float64)


def inside_stack(vox: torch.Tensor, stack_geometry: dict) -> torch.Tensor:
    r"""Whether each point (..., 3), in the stack's voxels, lies inside its
    volume: its in-plane field of view, and through-plane from half a slice
    thickness below its first slice to half a thickness above its last. Gaps
    between slices count as inside."""
    g = stack_geometry
    # Half a voxel in-plane; half a slice thickness through-plane, in voxels.
    thickness_vox = float(g["kernel_size_mm"].flatten()[-1]) / abs(float(g["mm_per_vox"].flatten()[-1]))
    half = torch.tensor([0.5, 0.5, 0.5 * thickness_vox], dtype=torch.float64)
    lo, hi = -half, g["size_vox"] - 1 + half
    return ((vox >= lo) & (vox <= hi)).all(dim=-1)


def stack_region(item: dict, stack: str, view: str, geometry: LatentGeometry) -> torch.Tensor:
    r"""Whether each voxel of `view` (on its cropped grid) has its centre inside
    `stack`'s volume (see `inside_stack`)."""
    lvox_to_vox = lvox_to_stack_vox(item, stack, geometry)
    centres = item["views"][view]["op"]["y_centres_lvox"].to(torch.float64)  # (n0, n1, n2, 3)
    vox = apply_affine(lvox_to_vox, centres.reshape(-1, 3).T).T.reshape(centres.shape)
    return inside_stack(vox, item["views"][stack]["geometry"])


class HeldoutProjection(NamedTuple):
    r"""What scoring a prediction of a held-out view needs (`heldout_projection`)."""

    project: Callable  # (1, 1, D, H, W) normalised latent -> the view's voxels, raw, zero off the mask
    target: torch.Tensor  # the held-out view on its cropped grid, raw intensities
    mask: torch.Tensor  # 1 on the scored voxels
    stack: str | None  # the stack bounding the scored region, if any


def heldout_projection(
    item: dict,
    view: str,
    geometry: LatentGeometry,
    lanczos_a: int = 3,
    region: str | None = None,
    device="cpu",
) -> HeldoutProjection | None:
    r"""The held-out `view`'s forward model, scored voxels and target, the same
    for every method scored on it.

    Scored on the voxels of `view` the forward model supports; with `region`
    (a view group, e.g. `sa`), only those inside the tallest of the item's
    stacks of that group in the geometry's views (`region_stack`,
    `stack_region`). None if the item lacks the view or no voxel is left.
    """
    if view not in item["views"]:
        return None
    heldout = item["views"][view]
    A = BatchOperator([{view: heldout["op"]}], geometry.size_lvox, lanczos_a=lanczos_a).to(device)
    mask = A.op(0, view).row_mask
    stack = None
    if region is not None:
        stack = region_stack(item, region, geometry.views)
        if stack is None:
            tqdm.write(f"{item['subject']}: no {region} stack to bound the scored region; skipped")
            return None
        mask = mask * stack_region(item, stack, view, geometry).to(mask)
    if mask.sum() == 0:
        if region is not None:
            tqdm.write(f"{item['subject']}: no {view} voxel inside {stack}; skipped")
        return None
    scale = item["norm_std"] / NORMALISED_STD

    def raw(values):
        return values * scale + item["norm_mean"]

    def project(x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return raw(A(x.float())[0][0][0, 0]) * mask

    target = raw(heldout["y"].to(device)[0, 0])
    return HeldoutProjection(project, target, mask, stack)


@torch.no_grad()
def evaluate_item(
    model: MRIVAE,
    item: dict,
    view: str,
    n_samples: int,
    conditioning_views,
    amp: bool = True,
    direct: bool = True,
    region: str | None = None,
):
    r"""Predictions of the held-out `view` of `item` from `conditioning_views`.

    Scored on the voxels of `view` the forward model supports; with `region`
    (a view group, e.g. `sa`), only those inside the tallest of the item's
    stacks of that group that the model was trained on (`region_stack`,
    `stack_region`), whichever views it is conditioned on.

    Returns the metrics row, the predictions, the projected samples' per-voxel
    standard deviation (None with fewer than 2 samples) and the target (raw
    intensities, on the view's cropped grid, zero off the mask), and the mask;
    or None if the item lacks the view or every conditioning view, or no voxel
    is left to score.
    """
    conditioning = {v: d for v, d in item["views"].items() if v in conditioning_views}
    if view not in item["views"] or not conditioning:
        return None
    projection = heldout_projection(
        item, view, model.geometry, lanczos_a=model.lanczos_a, region=region, device=model.device
    )
    if projection is None:
        return None
    project, target, mask, stack = projection
    n = mask.sum()
    cond_item = {**item, "views": conditioning}

    row = {
        "subject": item["subject"],
        "frame": item["frame"],
        "view": view,
        "conditioning_views": " ".join(conditioning),
        "n_voxels": int(n),
        **({"region_stack": stack} if stack is not None else {}),
        "norm_mean": item["norm_mean"],
        "norm_std": item["norm_std"],
    }
    predictions = {}
    std = None
    if n_samples > 0 and model.can_sample:
        samples = model.posterior([cond_item], n_samples=n_samples, amp=amp)  # (n, 1, 1, D, H, W)
        projected = torch.stack([project(sample) for sample in samples])
        per_sample = [masked_metrics(p, target, mask) for p in projected]
        for key in ("mse", "psnr", "ssim"):
            row[f"{key}_sample"] = float(np.mean([m[key] for m in per_sample]))
        # The forward model is linear: projecting the mean volume is the mean of the projections.
        predictions["sample_mean"] = projected.mean(0)
        if n_samples >= 2:
            # Also exact by linearity: the spread of the projected samples is
            # the posterior's spread in the view, sqrt(a_i^T Sigma a_i) per
            # voxel, which the latent's per-voxel std alone does not give.
            std = projected.std(0)
            error = (predictions["sample_mean"] - target).abs()
            row["mean_std"] = float(std[mask > 0.5].mean())
            row["spearman_std_error"] = masked_spearman(std, error, mask)
    if direct and model.direct is not None:
        predictions[model.direct_name] = project(model.predict_direct([cond_item], amp=amp))
    for name, prediction in predictions.items():
        metrics = masked_metrics(prediction, target, mask)
        for key in ("mse", "psnr", "ssim"):
            row[f"{key}_{name}"] = metrics[key]
        row["data_range"] = metrics["data_range"]
    return row, predictions, std, target, mask


def on_image_grid(values: torch.Tensor, heldout: dict) -> np.ndarray:
    r"""Places values on the view's cropped grid into its full image, zero elsewhere."""
    full = np.zeros(heldout["image_shape"], dtype=np.float32)
    x0, y0, z0 = heldout["crop_start"]
    n0, n1, n2 = values.shape
    full[x0 : x0 + n0, y0 : y0 + n1, z0 : z0 + n2] = values.cpu().numpy()
    return full


def save_on_image_grid(out_dir: str, item: dict, view: str, values: torch.Tensor, name: str):
    r"""Writes values on `view`'s cropped grid as <out_dir>/<view>_<name>.nii.gz,
    on the view's own image grid, with the frame in the header's toffset."""
    heldout = item["views"][view]
    nifti = nib.Nifti1Image(on_image_grid(values, heldout), affine=heldout["image_affine"].numpy())
    if item["frame"] is not None:
        nifti.header["toffset"] = item["frame"]
    nib.save(nifti, os.path.join(out_dir, f"{view}_{name}.nii.gz"))


def save_predictions(out_dir: str, item: dict, view: str, predictions: dict, target, mask, std=None):
    os.makedirs(out_dir, exist_ok=True)

    def save(values, name):
        save_on_image_grid(out_dir, item, view, values, name)

    save(target, "target")
    for name, values in predictions.items():
        save(values, f"pred_{name}")
    if std is not None:
        # Not pred_*, so mrisr-score never takes it for a prediction.
        save(std, "sample_std")
    save(mask, "mask")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="A training config; its `evaluate` section sets the options below")
    parser.add_argument("--checkpoint", help="Path to a checkpoint. Default: the config's best or last")
    parser.add_argument("--out_dir", help="Default: the config's, or <checkpoint_dir>/evaluation_<best|last>")
    parser.add_argument("--heldout_views", nargs="+", help="Default: the training config's data: heldout_views")
    parser.add_argument(
        "--conditioning_views",
        nargs="+",
        help="Conditioning sets: `all`, or comma-separated views or groups, e.g. `all sa la_2ch`. Default: all",
    )
    parser.add_argument("--data_dir", help="Defaults to the training config's, e.g. to evaluate on a separate test set")
    parser.add_argument("--subjects", nargs="+", help="Subject names. Default: the subjects of --split")
    parser.add_argument("--split", choices=["val", "train", "all"])
    parser.add_argument("--max_subjects", type=int)
    parser.add_argument("--frame", type=int, help="Default: a frame fixed per subject")
    parser.add_argument("--n_samples", type=int, help="Default 8; 0 (and only 0) for a supervised model")
    parser.add_argument("--seed", type=int, help="Samples are reseeded per subject and conditioning set. Default 0")
    parser.add_argument("--save_predictions", action="store_true", default=None)
    parser.add_argument(
        "--direct",
        action=argparse.BooleanOptionalAction,
        help="Score the direct predictor, if the model has one (--no-direct: the samples only). Default: on",
    )
    parser.add_argument(
        "--region",
        help="Score only inside the tallest stack of this view group, e.g. sa; none: everywhere. Default: none",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args(argv)
    n_samples_given = args.n_samples is not None

    # Command line, then the config's evaluate section, then its defaults.
    keys = (
        "out_dir", "data_dir", "split", "conditioning_views", "frame", "n_samples", "seed", "save_predictions",
        "direct", "region",
    )
    options_from_config(parser, args, "evaluate", keys, out_dir_prefix="evaluation")
    if args.region in ("none", "null"):
        args.region = None
    print(f"Checkpoint {args.checkpoint}; writing to {args.out_dir}")

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    geometry = LatentGeometry.from_config(state["geometry_config"])
    model = build_model(config, geometry)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    amp = config["train"]["amp"]

    heldout_views = args.heldout_views or config["data"]["heldout_views"]
    if not heldout_views:
        parser.error("No held-out views: pass --heldout_views or set data: heldout_views")
    trained_on = set(heldout_views) & set(geometry.views)
    if trained_on:
        parser.error(f"{sorted(trained_on)} were used in training, so they are not held out")
    if not model.can_sample:
        if n_samples_given and args.n_samples > 0:
            parser.error(f"A {model.direct_name} model has no posterior to sample: pass --n_samples 0")
        args.n_samples = 0
    direct = args.direct and model.direct is not None
    if args.n_samples < 0 or (args.n_samples == 0 and not direct):
        if model.direct is not None and not args.direct:
            parser.error("With direct off and no samples there is nothing to score: give n_samples, or turn direct on")
        parser.error("--n_samples must be positive, or 0 with a direct predictor")
    try:
        conditioning_sets = {spec: resolve_conditioning(spec, geometry.views) for spec in args.conditioning_views}
    except ValueError as error:
        parser.error(str(error))
    if args.region is not None and not any(view_group(v) == args.region for v in geometry.views):
        parser.error(f"--region `{args.region}` is no group of the views the model was trained on: {geometry.views}")

    data_dir = args.data_dir or config["data"]["data_dir"]
    if args.subjects is not None:
        names = args.subjects
    elif args.data_dir is not None and args.data_dir != config["data"]["data_dir"]:
        # A separate test set: every subject in it.
        names = None
    else:
        train_names, val_names = split_subjects(data_dir, geometry.views, seed=config["split_seed"])
        names = {"train": train_names, "val": val_names, "all": train_names + val_names}[args.split]
    if names is not None and args.max_subjects is not None:
        names = names[: args.max_subjects]

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for view in heldout_views:
        for spec, conditioning_views in conditioning_sets.items():
            # Every view is loaded, so the frame (chosen from the frames all of
            # them have) is the same for every conditioning set; only the
            # conditioning views set the normalisation.
            dataset = CMRObservationDataset(
                data_dir,
                geometry,
                subjects=names,
                extra_noise_std=config["data"]["extra_noise_std"],
                views=list(geometry.views) + [view],
                norm_views=conditioning_views,
                train=False,
                frame=args.frame,
            )
            if names is None and args.max_subjects is not None:
                dataset.subjects = dataset.subjects[: args.max_subjects]
            loader = DataLoader(dataset, batch_size=1, collate_fn=collate, num_workers=args.num_workers)
            for items in tqdm(loader, ncols=80, desc=f"{view} from {spec}"):
                if not items:
                    continue
                item = items[0]
                # A subject's samples depend on the seed, the subject and the
                # conditioning set only, not on which other subjects are run.
                with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                    torch.manual_seed(args.seed + zlib.crc32(f"{item['subject']}/{spec}".encode()))
                    result = evaluate_item(
                        model, item, view, args.n_samples, conditioning_views, amp=amp, direct=direct,
                        region=args.region,
                    )
                if result is None:
                    continue
                row, predictions, std, target, mask = result
                row = {"conditioning": conditioning_label(spec), **row}
                rows.append(row)
                if args.save_predictions:
                    subject_dir = os.path.join(args.out_dir, conditioning_label(spec), item["subject"])
                    save_predictions(subject_dir, item, view, predictions, target, mask, std=std)

    if not rows:
        print("No subject had both a held-out view and a view to condition on")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for view in heldout_views:
        for spec in conditioning_sets:
            selected = [r for r in rows if r["view"] == view and r["conditioning"] == conditioning_label(spec)]
            if not selected:
                continue
            summary = ", ".join(
                f"{key} {np.nanmean([r.get(key, np.nan) for r in selected]):.4g}"
                for key in fields
                if key.split("_")[0] in ("mse", "psnr", "ssim", "spearman")
            )
            print(f"{view} from {spec}: {len(selected)} subjects; mean {summary}")


if __name__ == "__main__":
    main()
