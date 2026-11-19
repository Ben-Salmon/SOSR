r"""Predicts the held-out view(s) with one method of an evaluation protocol.

    uv run mrisr-evaluate --protocol configs/evaluation/kaggle_cmr_test.yaml --method vae
        [--subjects ...] [--max_subjects N] [--out_dir <dir>]

For each subject and conditioning set of the protocol (see `protocol`), the
method's model is given those views only -- normalised from them alone -- and
its super-resolved volume is pushed through the held-out view's forward model.
With `n_samples` > 0 the prediction is the mean of that many posterior samples
(reseeded per subject and conditioning set from the protocol's seed, so a
subject's samples do not depend on which others are run), and their spread in
the view is saved too; with 0, it is the model's direct predictor (or the
supervised baseline's output).

Writes, in the protocol's out_dir (see `files`):

    <conditioning>/<subject>/<view>_pred_<method>.nii.gz
    <conditioning>/<subject>/<view>_std_<method>.nii.gz   with 2 or more samples
    <conditioning>/<subject>/<view>_target.nii.gz, <view>_mask.nii.gz
    runs/<method>.csv   per subject: frame, conditioning views, normalisation,
                        and with samples the average score of single samples

then scores the method (`score`). Run `mrisr-score` once every method and
baseline is done for the table comparing them.
"""

import argparse
import zlib

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import collate
from ..geometry import LatentGeometry
from ..metrics import masked_metrics
from ..model import build_model
from . import files, score
from .conditioning import conditioning_label, heldout_projection, item_conditioning
from .protocol import Method, Protocol, load_protocol, same_geometry


def load_method(protocol: Protocol, method: Method, device):
    r"""The method's model, checked against the protocol; and its training config."""
    state = torch.load(method.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    geometry = LatentGeometry.from_config(state["geometry_config"])
    if not same_geometry(geometry, protocol.geometry):
        raise SystemExit(f"{method.checkpoint}: its latent geometry is not the protocol's {protocol.geometry_path}")
    model = build_model(config, geometry)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    if model.lanczos_a != protocol.lanczos_a:
        raise SystemExit(f"{method.checkpoint}: lanczos_a {model.lanczos_a}, the protocol's {protocol.lanczos_a}")
    if method.n_samples > 0 and not model.can_sample:
        raise SystemExit(f"Method {method.name}: a {model.direct_name} model has no posterior: set n_samples 0")
    if method.n_samples == 0 and model.direct is None:
        raise SystemExit(f"Method {method.name}: the model has no direct predictor: set n_samples > 0")
    return model, config


@torch.no_grad()
def predict_item(model, item: dict, view: str, n_samples: int, conditioning_views, region=None, amp=True):
    r"""The method's prediction of held-out `view` of `item` from
    `conditioning_views`: (projection, prediction, std or None, row), or None
    if the item lacks the view or every conditioning view, or no voxel is
    left to score. The row holds the mean single-sample scores, with samples."""
    conditioning = {v: d for v, d in item["views"].items() if v in conditioning_views}
    if view not in item["views"] or not conditioning:
        return None
    projection = heldout_projection(
        item, view, model.geometry, lanczos_a=model.lanczos_a, region=region, device=model.device
    )
    if projection is None:
        return None
    cond_item = {**item, "views": conditioning}
    row, std = {}, None
    if n_samples > 0:
        samples = model.posterior([cond_item], n_samples=n_samples, amp=amp)  # (n, 1, 1, D, H, W)
        projected = torch.stack([projection.project(sample) for sample in samples])
        per_sample = [masked_metrics(p, projection.target, projection.mask) for p in projected]
        for key in ("mse", "psnr", "ssim"):
            row[f"{key}_sample"] = float(np.mean([m[key] for m in per_sample]))
        # The forward model is linear: projecting the mean volume is the mean of the projections.
        prediction = projected.mean(0)
        if n_samples >= 2:
            # Also exact by linearity: the spread of the projected samples is
            # the posterior's spread in the view, sqrt(a_i^T Sigma a_i) per
            # voxel, which the latent's per-voxel std alone does not give.
            std = projected.std(0) * projection.mask
    else:
        prediction = projection.project(model.predict_direct([cond_item], amp=amp))
    return projection, prediction, std, row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", required=True, help="An evaluation protocol yaml")
    parser.add_argument("--method", required=True, help="One of the protocol's methods")
    parser.add_argument("--subjects", nargs="+", help="Only these subjects (default: the protocol's)")
    parser.add_argument("--max_subjects", type=int, help="Only the first N subjects, e.g. for timing")
    parser.add_argument("--out_dir", help="Default: the protocol's, e.g. another for a timing run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args(argv)

    try:
        protocol = load_protocol(args.protocol)
    except ValueError as error:
        parser.error(str(error))
    if args.method not in protocol.methods:
        parser.error(f"`{args.method}` is not a method of {args.protocol}: {', '.join(protocol.methods)}")
    if args.out_dir is not None:
        protocol.out_dir = args.out_dir
    method = protocol.methods[args.method]
    device = torch.device(args.device)
    model, config = load_method(protocol, method, device)
    protocol.record()
    print(f"Method {method.name}: {method.checkpoint}, {method.n_samples or 'no'} samples; writing to {protocol.out_dir}")

    amp = config["train"]["amp"]
    rows = []
    for view in protocol.heldout_views:
        for spec, conditioning_views in protocol.conditioning_sets().items():
            label = conditioning_label(spec)
            dataset = protocol.dataset(
                view, spec, args.subjects, args.max_subjects, extra_noise_std=config["data"]["extra_noise_std"]
            )
            loader = DataLoader(dataset, batch_size=1, collate_fn=collate, num_workers=args.num_workers)
            for items in tqdm(loader, ncols=80, desc=f"{view} from {spec}"):
                if not items:
                    continue
                item = items[0]
                views = item_conditioning(spec, item, conditioning_views)
                with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                    torch.manual_seed(protocol.seed + zlib.crc32(f"{item['subject']}/{spec}".encode()))
                    result = predict_item(
                        model, item, view, method.n_samples, views, region=protocol.region, amp=amp
                    )
                if result is None:
                    continue
                projection, prediction, std, sample_row = result
                directory = files.subject_dir(protocol.out_dir, label, item["subject"])
                files.write_or_verify_references(directory, item, view, projection.target, projection.mask)
                files.save_on_image_grid(directory, item, view, prediction, files.prediction_name(method.name))
                if std is not None:
                    files.save_on_image_grid(directory, item, view, std, files.std_name(method.name))
                rows.append(
                    {
                        "conditioning": label,
                        "subject": item["subject"],
                        "frame": item["frame"],
                        "view": view,
                        "conditioning_views": " ".join(v for v in item["views"] if v in views),
                        "region_stack": projection.stack,
                        "n_voxels": int(projection.mask.sum()),
                        "norm_mean": item["norm_mean"],
                        "norm_std": item["norm_std"],
                        **sample_row,
                    }
                )

    if not rows:
        print("No subject had both a held-out view and a view to condition on")
        return
    run_path = f"{protocol.out_dir}/runs/{method.name}.csv"
    files.write_rows(run_path, rows)
    print(f"Wrote {len(rows)} predictions; {run_path}")
    score.print_summary(score.summarise(score.score_directory(protocol.out_dir, methods=[method.name])))


if __name__ == "__main__":
    main()
