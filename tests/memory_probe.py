r"""Peak GPU memory and time of training steps at full size, on real subjects.

    uv run python tests/memory_probe.py --config configs/kaggle_cmr.yaml \
        --n_filters 32 64 --batch_size 2

Loads a few training subjects (timing each item, which is what a DataLoader
worker does), then for each n_filters runs a few training steps and reports
peak memory and seconds per step. Use it to set `data: batch_size` and
`model: n_filters`, and to see whether data loading or the GPU is the
bottleneck.
"""

import argparse
import copy
import time

import torch

from mrisr_vae.data import CMRObservationDataset, split_subjects
from mrisr_vae.model import build_model
from mrisr_vae.train import load_config, load_geometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n_filters", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    geometry = load_geometry(config["geometry"])
    print(f"Latent grid {geometry.size_lvox}, views {geometry.views}")
    train_names, _ = split_subjects(config["data"]["data_dir"], geometry.views, seed=config["split_seed"])
    dataset = CMRObservationDataset(
        config["data"]["data_dir"],
        geometry,
        subjects=train_names[: 4 * args.batch_size],
        extra_noise_std=config["data"]["extra_noise_std"],
        train=True,
        view_drop_probability=0.0,  # every view: the most memory
    )
    items = []
    for i in range(len(dataset)):
        start = time.perf_counter()
        item = dataset[i]
        elapsed = time.perf_counter() - start
        if item is None:
            print(f"Item {i}: unusable")
            continue
        rows = sum(int(d["op"]["row_mask"].sum()) for d in item["views"].values())
        print(f"Item {i} ({item['subject']}): {elapsed:.2f} s to load, views {list(item['views'])}, {rows} observed voxels")
        items.append(item)
        if len(items) == args.batch_size:
            break
    if len(items) < args.batch_size:
        raise RuntimeError(f"Only {len(items)} usable items")

    device = torch.device("cuda")
    for n_filters in args.n_filters:
        model_config = copy.deepcopy(config)
        model_config["model"]["n_filters"] = n_filters
        model = build_model(model_config, geometry).to(device)
        n_parameters = sum(p.numel() for p in model.parameters())
        optimiser = torch.optim.Adamax(model.parameters(), lr=1e-6)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            times = []
            for _ in range(args.steps):
                torch.cuda.synchronize()
                start = time.perf_counter()
                # As in Trainer.train_epoch: the VAE's graph is freed before
                # the direct predictor's forward pass.
                out = model.elbo(items, amp=config["train"]["amp"])
                out["loss"].backward()
                inp, target = out["input"], out["x"].detach()
                nelbo = float(out["nelbo"])
                del out
                if model.direct is not None:
                    model.direct_loss(inp, target, amp=config["train"]["amp"]).backward()
                del inp, target
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(
                f"n_filters {n_filters} ({n_parameters / 1e6:.2f} M parameters, direct predictor "
                f"{'on' if model.direct is not None else 'off'}), batch {args.batch_size}: "
                f"peak {peak:.1f} GiB, {min(times):.2f} s/step (first {times[0]:.2f} s), "
                f"nelbo {nelbo:.4g}"
            )
            start = time.perf_counter()
            model.sample_prior(1)
            torch.cuda.synchronize()
            print(f"  prior sample: {time.perf_counter() - start:.2f} s")
        except torch.cuda.OutOfMemoryError:
            print(f"n_filters {n_filters}, batch {args.batch_size}: out of memory")
        del model, optimiser
        torch.cuda.empty_cache()
    print(f"GPU: {torch.cuda.get_device_name()}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")


if __name__ == "__main__":
    main()
