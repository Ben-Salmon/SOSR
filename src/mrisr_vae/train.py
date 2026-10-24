r"""Trains the VAE.

    uv run mrisr-train --config configs/kaggle_cmr.yaml [--resume]

Paths in the config are relative to the working directory, the repository
root in the SLURM scripts.
"""

import argparse
import copy
import os
import uuid

import torch
import yaml
from torch.utils.data import DataLoader

from .data import CMRObservationDataset, collate, split_subjects, subjects_with_views
from .geometry import LatentGeometry
from .likelihood import NOISE_MODES
from .model import MODEL_TYPES, build_model, load_pretrained_vae
from .trainer import SupervisedTrainer, Trainer

DEFAULTS = {
    # Latent grid and views: a diffusion-priors latent_image_geometry yaml.
    "geometry": None,
    "checkpoint_dir": None,
    "device": "cuda",
    "seed": 0,
    # Seed of the 90/10 subject split, as in diffusion-priors' build_datasets.
    "split_seed": 42,
    "data": {
        "data_dir": None,
        # Raw intensity units, by view. Enters the likelihood only (see
        # likelihood.GaussianLikelihood); no noise is added to the data.
        "extra_noise_std": None,
        # Each view is dropped from a training item's encoder input with this
        # probability, keeping at least one. It stays in the likelihood, so the
        # model learns to predict views it was not shown.
        "view_drop_probability": 0.3,
        # Views kept out of training and validation entirely, for evaluation
        # by mrisr-evaluate only. They must not be in the geometry's VIEWS,
        # which are the only views training ever loads.
        "heldout_views": [],
        # model: type: supervised only: the held-out view (one of
        # heldout_views) the baseline is trained to predict. Subjects without
        # it are skipped.
        "target_view": None,
        "batch_size": 2,
        "num_workers": 8,
    },
    "model": {
        # vae: the Ladder VAE (and its direct predictor). supervised: the
        # baseline, the direct predictor's UNet alone, trained against
        # data: target_view; z_dim and direct_predictor are then unused.
        "type": "vae",
        "n_layers": 8,
        "z_dim": 32,
        "n_filters": 32,
        "blocks_per_layer": 1,
        # `auto`: as many halvings as the latent grid allows (floor(log2(size))
        # - 1 per axis), placed as in COSDD's get_models. Or per layer, bottom
        # to top: 1 halves the resolution, e.g. [0, 1, 0, 1, 0, 1, 0, 1].
        "downsampling": "auto",
        "scale_initialisation": False,
        "checkpointed": True,
        # Encoder input is zero where coverage is below this fraction of its maximum.
        "coverage_threshold": 0.01,
        # Also train a UNet (same depth and width) to predict E[x | y] in one
        # pass, by regression onto the VAE's posterior samples (COSDD's direct
        # denoiser).
        "direct_predictor": True,
    },
    "likelihood": {
        # per_group: sigma set by extra_noise_std, as in diffusion-priors.
        # global: one sigma for every view, in normalised units, times a fixed
        # ratio per view (relative_std); extra_noise_std is not used.
        "noise": "per_group",
        # per_group: False keeps extra_noise_std as it is, True learns one
        # scale on it per view group (sa, la). global: whether sigma is learned
        # or stays at init_sigma.
        "learn_noise": False,
        "noise_lr": 0.01,
        "min_sigma": 0.001,
        # global only: the starting sigma, and the ratio of each view's sigma
        # to it, e.g. {la_2ch: 4}. Views not listed get 1.
        "init_sigma": 0.05,
        "relative_std": {},
    },
    "train": {
        "lr": 3.0e-4,
        "lr_warmup_steps": 1000,
        # The KL's weight rises linearly from 0 to 1 over this many optimiser
        # steps. 0 disables it.
        "kl_warmup_steps": 0,
        "grad_batches": 4,
        "clip": 1.0,
        # KL clamped below at this many nats per observed voxel for the first
        # free_bits_epochs epochs. 0 disables it.
        "free_bits": 0.0,
        "free_bits_epochs": 20,
        "max_epochs": 1000,
        # Stop (cleanly, resumable) after this many hours. None = no limit.
        "max_hours": None,
        "patience": 50,
        "plateau_patience": 20,
        "plateau_factor": 0.1,
        # val/nelbo or val/heldout_mse, or for the direct predictor
        # val/direct_mse or val/direct_heldout_mse (required with direct_only).
        # model: type: supervised: val/target_mse.
        "monitor": "val/nelbo",
        # A checkpoint whose VAE (Ladder VAE and noise) to start from. Its
        # direct predictor is not loaded: the new one starts afresh.
        "pretrained_vae": None,
        # Freeze the VAE (set pretrained_vae) and train only the direct
        # predictor, on its posterior samples. Early stopping as for the VAE,
        # with no learning-rate plateaus.
        "direct_only": False,
        "image_every": 5,
        # bfloat16 autocast for the network (never for the forward model)
        "amp": True,
    },
    "wandb": {
        "project": "mrisr-vae",
        "mode": "online",
    },
    # Used by mrisr-predict --config only; training ignores it.
    "predict": {
        # best or last: <checkpoint_dir>/<checkpoint>.pt
        "checkpoint": "best",
        # None: <checkpoint_dir>/predictions_<checkpoint>
        "out_dir": None,
        # None: data: data_dir
        "data_dir": None,
        # Subjects of data_dir to predict: val or train (the training split of
        # data_dir, by split_seed), or all (e.g. for a separate test set).
        "split": "val",
        # The frame to super-resolve in every subject; subjects without it
        # are skipped. None: the validation frame fixed per subject.
        "frame": None,
        # direct (the direct predictor, one pass), sampling (posterior
        # samples: their mean and std), or both.
        "method": "both",
        "n_samples": 8,
        "save_samples": False,
    },
    # Used by mrisr-evaluate --config only; training ignores it.
    "evaluate": {
        # best or last: <checkpoint_dir>/<checkpoint>.pt
        "checkpoint": "best",
        # None: <checkpoint_dir>/evaluation_<checkpoint>
        "out_dir": None,
        # None: data: data_dir. Another directory is a test set: all its subjects.
        "data_dir": None,
        # With the training data_dir only: val, train or all.
        "split": "val",
        # Each scored separately: all (every trained view), or comma-separated
        # views or view groups, e.g. sa, la_2ch, sa,la_2ch.
        "conditioning_views": ["all"],
        # None: a frame fixed per subject, the same for every model and run.
        "frame": None,
        # Posterior samples (VAE only); 2 or more also give the per-voxel std
        # and its correlation with the error. 0: direct predictor only.
        "n_samples": 8,
        # Samples are reseeded per subject and conditioning set from this.
        "seed": 0,
        # Write target, predictions, std and mask on the view's own grid.
        "save_predictions": False,
        # Score the direct predictor, if the model has one. False: the samples
        # only, so every metric is of the samples behind the std.
        "direct": True,
        # A view group, e.g. sa: score only the held-out voxels inside the
        # tallest stack of that group (highest index breaking ties), the same
        # voxels whatever the conditioning set. None: wherever the held-out
        # view's forward model is supported.
        "region": None,
    },
}


def load_config(path: str) -> dict:
    r"""Reads a config and fills in defaults, rejecting unknown options."""
    with open(path) as f:
        given = yaml.safe_load(f) or {}
    config = copy.deepcopy(DEFAULTS)
    for key, value in given.items():
        if key not in config:
            raise ValueError(f"`{key}` is not a valid configuration option")
        if isinstance(config[key], dict):
            for subkey, subvalue in (value or {}).items():
                if subkey not in config[key]:
                    raise ValueError(f"`{key}: {subkey}` is not a valid configuration option")
                config[key][subkey] = subvalue
        else:
            config[key] = value
    for key in ("geometry", "checkpoint_dir"):
        if config[key] is None:
            raise ValueError(f"Set `{key}` in the config")
    if config["data"]["data_dir"] is None:
        raise ValueError("Set `data: data_dir` in the config")
    if config["likelihood"]["noise"] not in NOISE_MODES:
        raise ValueError(f"`likelihood: noise` must be one of {', '.join(NOISE_MODES)}")
    if config["model"]["type"] == "vae":
        check_vae_options(config)
    elif config["model"]["type"] == "supervised":
        check_supervised_options(config)
    else:
        raise ValueError(f"`model: type` must be one of {', '.join(MODEL_TYPES)}")
    return config


def options_from_config(parser, args, section: str, keys, out_dir_prefix: str):
    r"""Fills the options `keys` of `args` left unset (None) on the command
    line from the `section` (predict or evaluate) of `args.config`, or from
    its defaults without a config.

    With a config, the checkpoint defaults to <checkpoint_dir>/<best|last>.pt
    and out_dir to <checkpoint_dir>/<out_dir_prefix>_<best|last>. Without one,
    --checkpoint and --out_dir must be given.
    """
    options = dict(DEFAULTS[section])
    if args.config is not None:
        run_config = load_config(args.config)
        options = dict(run_config[section])
        if options["checkpoint"] not in ("best", "last"):
            parser.error(f"`{section}: checkpoint` must be best or last")
        if args.checkpoint is None:
            args.checkpoint = os.path.join(run_config["checkpoint_dir"], f"{options['checkpoint']}.pt")
        if options["out_dir"] is None:
            options["out_dir"] = os.path.join(
                run_config["checkpoint_dir"], f"{out_dir_prefix}_{options['checkpoint']}"
            )
    for key in keys:
        if getattr(args, key) is None:
            setattr(args, key, options[key])
    if args.checkpoint is None or args.out_dir is None:
        parser.error("Give --config, or --checkpoint and --out_dir")
    if args.split not in ("val", "train", "all"):
        parser.error(f"`{section}: split` must be val, train or all")
    if args.frame is not None and args.frame < 0:
        parser.error("The frame must not be negative")


def check_vae_options(config: dict):
    train_cfg = config["train"]
    if config["data"]["target_view"] is not None:
        raise ValueError("`data: target_view` is for `model: type: supervised` only")
    vae_monitors, direct_monitors = ("val/nelbo", "val/heldout_mse"), ("val/direct_mse", "val/direct_heldout_mse")
    if train_cfg["monitor"] not in vae_monitors + direct_monitors:
        raise ValueError(f"`train: monitor` must be one of {', '.join(vae_monitors + direct_monitors)}")
    if train_cfg["monitor"] in direct_monitors and not config["model"]["direct_predictor"]:
        raise ValueError(f"`train: monitor: {train_cfg['monitor']}` needs `model: direct_predictor: true`")
    if train_cfg["direct_only"]:
        if not config["model"]["direct_predictor"]:
            raise ValueError("`train: direct_only` needs `model: direct_predictor: true`")
        if train_cfg["pretrained_vae"] is None:
            raise ValueError("`train: direct_only` needs `train: pretrained_vae`, the VAE to train it on")
        if train_cfg["monitor"] not in direct_monitors:
            # The frozen VAE's validation values do not change.
            raise ValueError(f"With `train: direct_only`, `train: monitor` must be one of {', '.join(direct_monitors)}")


def check_supervised_options(config: dict):
    r"""The supervised baseline needs a held-out view to train against, and
    has none of the VAE's options."""
    target = config["data"]["target_view"]
    if target is None:
        raise ValueError("`model: type: supervised` needs `data: target_view`, the view to predict")
    if target not in config["data"]["heldout_views"]:
        # Held-out views are never among the views the encoder is shown (see check_heldout_views).
        raise ValueError(f"`data: target_view` {target} must be one of `data: heldout_views`")
    if config["train"]["monitor"] != "val/target_mse":
        raise ValueError("With `model: type: supervised`, `train: monitor` must be val/target_mse")
    for key in ("direct_only", "pretrained_vae"):
        if config["train"][key]:
            raise ValueError(f"`train: {key}` is for `model: type: vae` only")


def load_geometry(path: str) -> LatentGeometry:
    with open(path) as f:
        return LatentGeometry.from_config(yaml.safe_load(f))


def check_relative_std(config: dict, geometry: LatentGeometry):
    r"""Rejects ratios for views that neither training nor evaluation uses,
    which would most likely be typos."""
    views = {*geometry.views, *config["data"]["heldout_views"]}
    unknown = set(config["likelihood"]["relative_std"] or {}) - views
    if unknown:
        raise ValueError(f"`likelihood: relative_std` names unknown views {sorted(unknown)}; views: {sorted(views)}")


def check_heldout_views(config: dict, geometry: LatentGeometry):
    r"""Refuses a config whose held-out views would be trained on."""
    overlap = set(config["data"]["heldout_views"]) & set(geometry.views)
    if overlap:
        raise ValueError(
            f"Held-out views {sorted(overlap)} are in the geometry's VIEWS, so training "
            "would load them. Remove them from VIEWS or from data: heldout_views."
        )


def check_pretrained_vae(config: dict):
    r"""Refuses a pretrained VAE in this run's checkpoint_dir, whose best.pt and
    last.pt the run would overwrite."""
    pretrained = config["train"]["pretrained_vae"]
    if pretrained is None:
        return
    if not os.path.isfile(pretrained):
        raise ValueError(f"`train: pretrained_vae` {pretrained} does not exist")
    if os.path.realpath(os.path.dirname(pretrained)) == os.path.realpath(config["checkpoint_dir"]):
        raise ValueError(
            f"`train: pretrained_vae` {pretrained} is in checkpoint_dir, which this run would "
            "overwrite. Give the run its own checkpoint_dir."
        )


def make_loader(dataset, batch_size, num_workers, shuffle, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Carry on from <checkpoint_dir>/last.pt. Without one this starts afresh, "
        "so it is safe to leave in a submission script.",
    )
    parser.add_argument("--checkpoint_dir", help="Overrides the config's checkpoint_dir")
    parser.add_argument("--max_epochs", type=int, help="Overrides train: max_epochs")
    parser.add_argument(
        "--max_subjects",
        type=int,
        help="Train on the first N training subjects and validate on N // 4 (at least 1), for smoke tests",
    )
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], help="Overrides wandb: mode")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.checkpoint_dir is not None:
        config["checkpoint_dir"] = args.checkpoint_dir
    if args.max_epochs is not None:
        config["train"]["max_epochs"] = args.max_epochs
    if args.wandb_mode is not None:
        config["wandb"]["mode"] = args.wandb_mode

    torch.manual_seed(config["seed"])
    geometry = load_geometry(config["geometry"])
    check_heldout_views(config, geometry)
    check_relative_std(config, geometry)
    check_pretrained_vae(config)
    print(f"Latent grid: {geometry.size_lvox} voxels of {float(geometry.su_per_lvox):.3g} su; views {geometry.views}")
    print(f"Never loaded in training or validation: {config['data']['heldout_views']}")

    data_cfg = config["data"]
    train_names, val_names = split_subjects(data_cfg["data_dir"], geometry.views, seed=config["split_seed"])
    supervised = config["model"]["type"] == "supervised"
    target_views = [data_cfg["target_view"]] if supervised else None
    if supervised:
        # The same split as the VAE's, less the subjects without the target.
        for split, names in (("training", train_names), ("validation", val_names)):
            kept = subjects_with_views(data_cfg["data_dir"], names, target_views)
            print(f"{len(names) - len(kept)} of {len(names)} {split} subjects have no {target_views[0]}; skipping them")
            names[:] = kept
        print(f"Training against {target_views[0]}, which the network is never shown")
    if args.max_subjects is not None:
        train_names = train_names[: args.max_subjects]
        val_names = val_names[: max(1, args.max_subjects // 4)]
    print(f"{len(train_names)} training and {len(val_names)} validation subjects")

    train_set = CMRObservationDataset(
        data_cfg["data_dir"],
        geometry,
        subjects=train_names,
        extra_noise_std=data_cfg["extra_noise_std"],
        train=True,
        view_drop_probability=data_cfg["view_drop_probability"],
        target_views=target_views,
    )
    val_set = CMRObservationDataset(
        data_cfg["data_dir"],
        geometry,
        subjects=val_names,
        extra_noise_std=data_cfg["extra_noise_std"],
        train=False,
        target_views=target_views,
    )
    device = torch.device(config["device"])
    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_set, data_cfg["batch_size"], data_cfg["num_workers"], True, pin_memory)
    val_loader = make_loader(val_set, data_cfg["batch_size"], data_cfg["num_workers"], False, pin_memory)

    model = build_model(config, geometry)
    if config["train"]["pretrained_vae"] is not None:
        # On --resume, last.pt below replaces these weights with its own.
        load_pretrained_vae(model, config["train"]["pretrained_vae"])
        print(f"VAE from {config['train']['pretrained_vae']}; direct predictor initialised afresh")
    if config["train"]["direct_only"]:
        print("Training the direct predictor only; the VAE is frozen")
    model = model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"{n_parameters / 1e6:.2f} M parameters")
    halvings = [sum(layer[i] for layer in model.downsampling) for i in range(3)]
    print(f"Downsampling per layer (bottom to top): {model.downsampling}; {halvings} halvings per axis")

    if supervised:
        trainer = SupervisedTrainer(
            model,
            train_loader,
            val_loader,
            train_config=config["train"],
            checkpoint_dir=config["checkpoint_dir"],
            target_view=target_views[0],
            config=config,
        )
    else:
        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            train_config=config["train"],
            noise_lr=config["likelihood"]["noise_lr"],
            checkpoint_dir=config["checkpoint_dir"],
            config=config,
        )
    last = os.path.join(config["checkpoint_dir"], "last.pt")
    if args.resume and os.path.exists(last):
        trainer.load(last)
        print(f"Resuming after {trainer.epoch} completed epoch(s)")

    import wandb

    if trainer.wandb_id is None:
        # Our own id rather than wandb.util.generate_id, which newer wandb
        # versions no longer have. Saved in the checkpoints, so --resume
        # continues the same run.
        trainer.wandb_id = uuid.uuid4().hex[:8]
    run = wandb.init(
        project=config["wandb"]["project"],
        id=trainer.wandb_id,
        resume="allow",
        mode=config["wandb"]["mode"],
        dir=config["checkpoint_dir"],
        config=config,
    )
    trainer.run = run
    try:
        trainer.fit()
    finally:
        run.finish()


if __name__ == "__main__":
    main()
