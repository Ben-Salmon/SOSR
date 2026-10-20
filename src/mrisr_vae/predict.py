r"""Super-resolves subjects with a trained VAE.

    uv run mrisr-predict --config configs/kaggle_cmr.yaml [options]
    uv run mrisr-predict --checkpoint <checkpoint_dir>/best.pt --out_dir <dir> [options]

With --config, the config's `predict` section (see `train.DEFAULTS`) sets the
checkpoint (best or last, in its checkpoint_dir), out_dir, data_dir, split,
frame, method, n_samples and save_samples. Options given on the command line
override it. Subjects without the requested frame are skipped and listed at
the end.

For each subject writes, in <out_dir>/<subject>/:
    sr_direct.nii.gz     the direct predictor's E[x | y], in one pass (method direct or both)
    sr_supervised.nii.gz the supervised baseline's estimate, in its place
    sr_mean.nii.gz       mean of the posterior samples, the MMSE estimate (method sampling or both)
    sr_std.nii.gz        their per-voxel standard deviation (with 2+ samples)
    sr_sample_<i>.nii.gz each posterior sample, with --save_samples
Method both on a model without a direct predictor writes only the samples. The
supervised baseline (`model: type: supervised`) has no posterior: it writes
sr_supervised only, its one estimate, so its files never overwrite a VAE's.

The volumes are on the latent grid, with its vox -> mm affine, and the frame
in the header's toffset, as diffusion-priors writes them. By default they
are rescaled to the subject's raw intensities.
"""

import argparse
import os

import nibabel as nib
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import NORMALISED_STD, CMRObservationDataset, collate, split_subjects
from .forward_model import invert_affine
from .geometry import LatentGeometry
from .model import build_model
from .train import options_from_config


def save_volume(volume, affine, frame, path):
    nifti = nib.Nifti1Image(volume.numpy(), affine=affine.numpy())
    if frame is not None:
        nifti.header["toffset"] = frame
    nib.save(nifti, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="A training config; its `predict` section sets the options below")
    parser.add_argument("--checkpoint", help="Path to a checkpoint. Default: the config's best or last")
    parser.add_argument("--out_dir", help="Default: the config's, or <checkpoint_dir>/predictions_<best|last>")
    parser.add_argument("--data_dir", help="Default: the config's, or the training data's")
    parser.add_argument("--subjects", nargs="+", help="Subject names. Default: the subjects of --split")
    parser.add_argument("--split", choices=["val", "train", "all"])
    parser.add_argument("--max_subjects", type=int)
    parser.add_argument("--views", nargs="+", help="Condition on these views only, e.g. the short-axis stacks")
    parser.add_argument(
        "--frame", type=int, help="Default: the config's, or the validation frame fixed per subject"
    )
    parser.add_argument("--method", choices=["direct", "sampling", "both"])
    parser.add_argument("--n_samples", type=int)
    parser.add_argument("--save_samples", action="store_true", default=None)
    parser.add_argument("--normalised", action="store_true", help="Keep normalised intensities")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args(argv)

    # Command line, then the config's predict section, then its defaults.
    keys = ("out_dir", "data_dir", "split", "frame", "method", "n_samples", "save_samples")
    options_from_config(parser, args, "predict", keys, out_dir_prefix="predictions")
    if args.method not in ("direct", "sampling", "both"):
        parser.error("`predict: method` must be direct, sampling or both")
    sampling = args.method in ("sampling", "both")
    if args.method == "sampling" and args.n_samples < 1:
        parser.error(f"Method {args.method} needs n_samples of at least 1")

    device = torch.device(args.device)
    print(f"Checkpoint {args.checkpoint}; writing to {args.out_dir}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    geometry = LatentGeometry.from_config(state["geometry_config"])
    model = build_model(config, geometry)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    amp = config["train"]["amp"]
    if args.method == "direct" and model.direct is None:
        parser.error("Method direct needs a model with a direct predictor")
    if not model.can_sample:
        # The supervised baseline: one estimate, no posterior.
        if args.method == "sampling":
            parser.error(f"A {model.direct_name} model has no posterior to sample: use method direct")
        if args.method == "both":
            print(f"A {model.direct_name} model has no posterior to sample; writing sr_{model.direct_name} only")
        sampling = False
    direct = args.method in ("direct", "both") and model.direct is not None
    if sampling and args.n_samples < 1:
        parser.error(f"Method {args.method} needs n_samples of at least 1")
    if args.views is not None and not set(args.views) <= set(geometry.views):
        parser.error(f"--views must be among the views the model was trained on: {geometry.views}")

    data_dir = args.data_dir or config["data"]["data_dir"]
    if args.subjects is not None:
        names = args.subjects
    else:
        train_names, val_names = split_subjects(data_dir, geometry.views, seed=config["split_seed"])
        names = {"train": train_names, "val": val_names, "all": train_names + val_names}[args.split]
    if args.max_subjects is not None:
        names = names[: args.max_subjects]

    dataset = CMRObservationDataset(
        data_dir,
        geometry,
        subjects=names,
        extra_noise_std=config["data"]["extra_noise_std"],
        views=args.views,
        train=False,
        frame=args.frame,
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate, num_workers=args.num_workers)

    print(f"{len(names)} subjects; frame {'fixed per subject' if args.frame is None else args.frame}")
    done = set()
    for items in tqdm(loader, ncols=80):
        if not items:
            continue
        item = items[0]
        done.add(item["subject"])

        def denormalise(x):
            return x if args.normalised else x * item["norm_std"] / NORMALISED_STD + item["norm_mean"]

        su_to_mm = invert_affine(item["mm_to_su_affine"].to(torch.float32))
        lvox_to_mm = su_to_mm @ geometry.lvox_to_su_affine
        out_dir = os.path.join(args.out_dir, item["subject"])
        os.makedirs(out_dir, exist_ok=True)
        if direct:
            prediction = denormalise(model.predict_direct(items, amp=amp)[0, 0].cpu())
            # sr_direct for the VAE's direct predictor, sr_supervised for the baseline.
            path = os.path.join(out_dir, f"sr_{model.direct_name}.nii.gz")
            save_volume(prediction, lvox_to_mm, item["frame"], path)
        if not sampling:
            continue
        samples = denormalise(model.posterior(items, n_samples=args.n_samples, amp=amp)[:, 0, 0].cpu())  # (n, D, H, W)
        save_volume(samples.mean(0), lvox_to_mm, item["frame"], os.path.join(out_dir, "sr_mean.nii.gz"))
        if args.n_samples > 1:
            save_volume(samples.std(0), lvox_to_mm, item["frame"], os.path.join(out_dir, "sr_std.nii.gz"))
        if args.save_samples:
            for i, sample in enumerate(samples):
                save_volume(sample, lvox_to_mm, item["frame"], os.path.join(out_dir, f"sr_sample_{i}.nii.gz"))

    skipped = sorted(set(names) - done)
    print(f"Wrote {len(done)} of {len(names)} subjects to {args.out_dir}")
    if skipped:
        # Reasons were printed as each was loaded: a missing frame, or no usable view.
        print(f"Skipped {len(skipped)}: {' '.join(skipped)}")


if __name__ == "__main__":
    main()
