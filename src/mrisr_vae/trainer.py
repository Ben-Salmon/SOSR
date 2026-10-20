r"""Training loop: Adamax with warm-up, early stopping and resumable checkpoints."""

import math
import os
import time
from collections import defaultdict

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from .model import MRIVAE, SupervisedPredictor
from .visualise import volume_images


class Trainer:
    r"""Trains an `MRIVAE` by maximising the ELBO.

    Validation each epoch reports the negative ELBO (with fixed frames, no view
    dropout and a fixed random seed, so epochs are comparable) and the
    held-out-view error. `monitor` picks which one decides the best checkpoint,
    the learning-rate plateaus and early stopping.

    Checkpoints in `checkpoint_dir`: `last.pt` after every epoch, for
    `--resume`, and `best.pt` whenever the monitored value improves.

    With `direct_only` in `train_config`, the VAE is frozen (load a trained
    one first) and only the direct predictor is trained, on its posterior
    samples. Early stopping is as for the VAE, on a direct-predictor metric,
    but there is no learning-rate plateau schedule.

    Arguments:
        loaders: Iterables of batches, i.e. lists of dataset items.
        train_config: The `train` section of the config.
        noise_lr: Learning rate of the likelihood's noise parameters.
        run: A wandb run, or None not to log.
        config: The full config and the geometry config, saved with checkpoints.
    """

    def __init__(
        self,
        model: MRIVAE,
        train_loader,
        val_loader,
        train_config: dict,
        noise_lr: float,
        checkpoint_dir: str,
        run=None,
        config: dict | None = None,
    ):
        self._init_state(model, train_loader, val_loader, train_config, checkpoint_dir, run, config)
        self.direct_only = train_config.get("direct_only", False)
        if self.direct_only:
            if model.direct is None:
                raise ValueError("direct_only needs a model with a direct predictor")
            model.lvae.requires_grad_(False)
            model.likelihood.requires_grad_(False)

        self.optimiser = optim.Adamax(
            [
                {"params": model.lvae.parameters(), "lr": train_config["lr"]},
                {"params": model.likelihood.parameters(), "lr": noise_lr},
            ]
        )
        self.scheduler = self._plateau(self.optimiser)
        # The direct predictor has its own optimiser and schedule, as in COSDD:
        # it is fitted to the VAE, not part of it.
        self.direct_optimiser = self.direct_scheduler = None
        if model.direct is not None:
            self.direct_optimiser = optim.Adamax(model.direct.parameters(), lr=train_config["lr"])

    def _init_state(self, model, train_loader, val_loader, train_config, checkpoint_dir, run, config):
        r"""What every trainer has: its data, settings, and the counters that
        checkpoints save."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_config
        self.checkpoint_dir = checkpoint_dir
        self.run = run
        self.config = config
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.epoch = 0
        self.n_steps = 0
        self.best = math.inf
        self.bad_epochs = 0
        self.wandb_id = None

    def _plateau(self, optimiser):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode="min",
            factor=self.cfg["plateau_factor"],
            patience=self.cfg["plateau_patience"],
        )

    @property
    def optimisers(self) -> list:
        return [o for o in (self.optimiser, self.direct_optimiser) if o is not None]

    @property
    def amp(self) -> bool:
        return self.cfg["amp"]

    def in_warmup(self) -> bool:
        r"""Whether the learning rate or the KL weight is still ramping up. With
        `direct_only` there is no KL in the loss, so only the learning rate."""
        if self.direct_only:
            return self.n_steps < self.cfg["lr_warmup_steps"]
        return self.n_steps < max(self.cfg["lr_warmup_steps"], self.cfg["kl_warmup_steps"])

    def kl_weight(self) -> float:
        r"""The KL's weight in the loss: from 0 to 1, linearly over the first
        `kl_warmup_steps` optimiser steps (Sønderby et al., 2016), so the
        latents are put to use before the KL pushes them back to the prior."""
        warmup = self.cfg["kl_warmup_steps"]
        return min(1.0, self.n_steps / warmup) if warmup > 0 else 1.0

    def _optimiser_step(self, optimiser):
        r"""Steps one optimiser, with its learning rates ramped linearly over the
        warm-up. Adamax's first steps move every weight by the full learning rate
        in the direction of its gradient's sign (see COSDD's Hub). Gradients are
        clipped by the norm of this optimiser's parameters alone."""
        warmup = self.cfg["lr_warmup_steps"]
        if self.n_steps <= warmup:
            factor = min(1.0, (self.n_steps + 1) / max(warmup, 1))
            for group in optimiser.param_groups:
                group.setdefault("warmup_base_lr", group["lr"])
                group["lr"] = group["warmup_base_lr"] * factor
        if self.cfg["clip"] is not None:
            params = [p for group in optimiser.param_groups for p in group["params"]]
            clip_grad_norm_(params, self.cfg["clip"])
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)

    def train_epoch(self) -> dict:
        r"""One pass over the training data.

        Per batch, in COSDD's order: the VAE's forward and backward pass, which
        frees its graph, its optimiser step when gradients have accumulated,
        and only then the direct predictor's forward and backward pass and
        step, on the encoder input and the (detached) posterior sample. The two
        networks' activations are never in memory together, so the direct
        predictor adds its own peak, not the VAE's on top.
        """
        if self.direct_only:
            return self.train_direct_epoch()
        self.model.train()
        grad_batches = self.cfg["grad_batches"]
        free_bits = self.cfg["free_bits"] if self.epoch < self.cfg["free_bits_epochs"] else 0.0
        direct = self.model.direct is not None
        sums = defaultdict(float)
        n_batches = 0
        n_direct = 0
        accumulated = 0
        for optimiser in self.optimisers:
            optimiser.zero_grad(set_to_none=True)
        for items in tqdm(self.train_loader, ncols=80, desc=f"Epoch {self.epoch}", leave=False):
            if not items:
                continue
            out = self.model.elbo(items, amp=self.amp, free_bits=free_bits, kl_weight=self.kl_weight())
            if not torch.isfinite(out["loss"]):
                print(f"Non-finite loss for {[item['subject'] for item in items]}; skipping the batch")
                continue
            (out["loss"] / grad_batches).backward()
            for key in ("loss", "nelbo", "nll", "kl"):
                sums[key] += out[key].item()
            for i, kl in enumerate(out["kl_layers"].detach().tolist()):
                sums[f"kl_layer_{i}"] += kl
            n_batches += 1
            # Only what the direct predictor needs outlives the VAE's graph.
            inp, target = out["input"], out["x"].detach()
            del out

            accumulated += 1
            step = accumulated == grad_batches
            if step:
                self._optimiser_step(self.optimiser)

            if direct:
                direct_mse = self.model.direct_loss(inp, target, amp=self.amp)
                if torch.isfinite(direct_mse):
                    (direct_mse / grad_batches).backward()
                    sums["direct_mse"] += direct_mse.item()
                    n_direct += 1
                del inp, target, direct_mse
                if step:
                    self._optimiser_step(self.direct_optimiser)

            if step:
                self.n_steps += 1
                accumulated = 0
        if accumulated:
            for optimiser in self.optimisers:
                self._optimiser_step(optimiser)
            self.n_steps += 1
        metrics = {f"train/{key}": value / max(n_batches, 1) for key, value in sums.items()}
        if direct:
            metrics["train/direct_mse"] = sums["direct_mse"] / max(n_direct, 1)
        return metrics

    def train_direct_epoch(self) -> dict:
        r"""One pass that trains only the direct predictor (`direct_only`), on
        one posterior sample of the frozen VAE per observation. The encoder
        input has view dropout, as in `train_epoch`."""
        self.model.train()
        grad_batches = self.cfg["grad_batches"]
        mse_sum = 0.0
        n_batches = 0
        accumulated = 0
        self.direct_optimiser.zero_grad(set_to_none=True)
        for items in tqdm(self.train_loader, ncols=80, desc=f"Epoch {self.epoch}", leave=False):
            if not items:
                continue
            inp, target = self.model.direct_targets(items, amp=self.amp)
            direct_mse = self.model.direct_loss(inp, target, amp=self.amp)
            del inp, target
            if not torch.isfinite(direct_mse):
                print(f"Non-finite direct loss for {[item['subject'] for item in items]}; skipping the batch")
                continue
            (direct_mse / grad_batches).backward()
            mse_sum += direct_mse.item()
            n_batches += 1
            del direct_mse

            accumulated += 1
            if accumulated == grad_batches:
                self._optimiser_step(self.direct_optimiser)
                self.n_steps += 1
                accumulated = 0
        if accumulated:
            self._optimiser_step(self.direct_optimiser)
            self.n_steps += 1
        return {"train/direct_mse": mse_sum / max(n_batches, 1)}

    @torch.no_grad()
    def validate(self) -> tuple[dict, list | None]:
        r"""Validation metrics, and the first validation batch for images."""
        self.model.eval()
        device = self.model.device
        nelbo_sum = 0.0
        rows = 0.0
        heldout_total = 0.0
        heldout_count = 0
        direct = self.model.direct is not None
        direct_mse_sum = 0.0
        direct_batches = 0
        direct_heldout_total = 0.0
        first = None
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(0)
            for items in self.val_loader:
                if not items:
                    continue
                if first is None:
                    first = items
                out = self.model.elbo(items, amp=self.amp)
                nelbo_sum += float(out["nelbo"]) * float(out["n_rows"])
                rows += float(out["n_rows"])
                total, count = self.model.heldout_error(items, amp=self.amp)
                heldout_total += float(total)
                heldout_count += count
                if direct:
                    direct_mse_sum += float(self.model.direct_loss(out["input"], out["x"], amp=self.amp))
                    direct_batches += 1
                    total, _ = self.model.heldout_error(items, amp=self.amp, direct=True)
                    direct_heldout_total += float(total)
        metrics = {
            "val/nelbo": nelbo_sum / rows if rows else math.nan,
            "val/heldout_mse": heldout_total / heldout_count if heldout_count else math.nan,
        }
        if direct:
            # Against single posterior samples, so it includes their spread: it
            # tracks the predictor's progress, not its error against the mean.
            metrics["val/direct_mse"] = direct_mse_sum / direct_batches if direct_batches else math.nan
            metrics["val/direct_heldout_mse"] = (
                direct_heldout_total / heldout_count if heldout_count else math.nan
            )
        return metrics, first

    @torch.no_grad()
    def validation_images(self, items: list) -> dict:
        r"""Central slices of the encoder input, two posterior samples, a prior
        sample and the direct prediction, for the first observation of `items`."""
        self.model.eval()
        items = items[:1]
        A, ys, sigma_meas, sigma_extra = self.model.prepare(items)
        inp = self.model.encoder_input(A, ys, sigma_meas, sigma_extra)
        volumes = {
            "input/backprojection": inp[0, 0],
            "input/coverage": inp[0, 1],
            "posterior/sample_1": self.model.decode(inp, amp=self.amp)[0][0, 0],
            "posterior/sample_2": self.model.decode(inp, amp=self.amp)[0][0, 0],
            "prior/sample": self.model.sample_prior(1, amp=self.amp)[0, 0],
        }
        if self.model.direct is not None:
            volumes["direct/prediction"] = self.model.direct_predict(inp, amp=self.amp)[0, 0]
        return volume_images(volumes)

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "direct_optimiser": None if self.direct_optimiser is None else self.direct_optimiser.state_dict(),
            "epoch": self.epoch,
            "n_steps": self.n_steps,
            "best": self.best,
            "bad_epochs": self.bad_epochs,
            "wandb_id": self.wandb_id,
            "config": self.config,
            "geometry_config": self.model.geometry.config,
        }

    def save(self, name: str):
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(self.state_dict(), path + ".tmp")
        os.replace(path + ".tmp", path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.model.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimiser.load_state_dict(state["optimiser"])
        self.scheduler.load_state_dict(state["scheduler"])
        if self.direct_optimiser is not None:
            self.direct_optimiser.load_state_dict(state["direct_optimiser"])
        self.epoch = state["epoch"]
        self.n_steps = state["n_steps"]
        self.best = state["best"]
        self.bad_epochs = state["bad_epochs"]
        self.wandb_id = state["wandb_id"]

    @property
    def uses_plateaus(self) -> bool:
        r"""Whether the learning rate drops on plateaus of the monitored value."""
        return not self.direct_only

    def schedule_log(self) -> dict:
        r"""The schedules' current values, logged every epoch."""
        return {
            "lr": (self.direct_optimiser if self.direct_only else self.optimiser).param_groups[0]["lr"],
            "kl_weight": self.kl_weight(),
            **self.model.likelihood.log_values(),
        }

    def summary(self, log: dict, metrics: dict) -> str:
        r"""The line printed after each epoch."""
        monitor = self.cfg["monitor"]
        return (
            f"Epoch {self.epoch}: train nelbo {log.get('train/nelbo', math.nan):.5g}, "
            f"val nelbo {metrics['val/nelbo']:.5g}, held-out mse {metrics['val/heldout_mse']:.5g}"
            + ("" if monitor in ("val/nelbo", "val/heldout_mse") else f", {monitor} {metrics[monitor]:.5g}")
        )

    def fit(self):
        start = time.time()
        monitor = self.cfg["monitor"]
        while self.epoch < self.cfg["max_epochs"]:
            log = {"epoch": self.epoch}
            log.update(self.train_epoch())
            metrics, first = self.validate()
            log.update(metrics)
            log["optimiser_steps"] = self.n_steps
            log.update(self.schedule_log())

            value = metrics[monitor]
            improved = False
            # Schedules that watch validation wait for the warm-ups to end, so
            # that values from before the learning rate and the KL weight are at
            # full strength set neither the best value nor the patience counts.
            if not self.in_warmup() and math.isfinite(value):
                if self.uses_plateaus:
                    self.scheduler.step(value)
                if value < self.best:
                    self.best = value
                    self.bad_epochs = 0
                    improved = True
                else:
                    self.bad_epochs += 1

            if self.run is not None:
                if first is not None and (self.epoch + 1) % self.cfg["image_every"] == 0:
                    import wandb

                    for name, image in self.validation_images(first).items():
                        log[name] = wandb.Image(image)
                self.run.log(log)
            print(self.summary(log, metrics))

            self.epoch += 1
            if improved:
                self.save("best.pt")
            self.save("last.pt")
            if self.bad_epochs >= self.cfg["patience"]:
                print(f"No improvement in {monitor} for {self.bad_epochs} epochs; stopping")
                break
            if self.cfg["max_hours"] is not None and time.time() - start > self.cfg["max_hours"] * 3600:
                print(f"Reached max_hours ({self.cfg['max_hours']}); stopping. Resume with --resume")
                break


class SupervisedTrainer(Trainer):
    r"""Trains a `SupervisedPredictor` against `target_view`, a view the
    encoder is never shown, by mean squared error through its forward model.

    As `Trainer` wherever the VAE's training carries over: Adamax with the
    learning-rate warm-up, gradient accumulation over `grad_batches`,
    clipping, learning-rate plateaus, early stopping on `monitor`
    (`val/target_mse`), `max_hours`, and the same checkpoints. There is no KL,
    so only the learning-rate warm-up holds back the schedules.
    """

    def __init__(
        self,
        model: SupervisedPredictor,
        train_loader,
        val_loader,
        train_config: dict,
        checkpoint_dir: str,
        target_view: str,
        run=None,
        config: dict | None = None,
    ):
        self._init_state(model, train_loader, val_loader, train_config, checkpoint_dir, run, config)
        self.target_view = target_view
        self.direct_only = False
        self.optimiser = optim.Adamax(model.direct.parameters(), lr=train_config["lr"])
        self.scheduler = self._plateau(self.optimiser)
        self.direct_optimiser = self.direct_scheduler = None

    def in_warmup(self) -> bool:
        return self.n_steps < self.cfg["lr_warmup_steps"]

    def train_epoch(self) -> dict:
        r"""One pass over the training data: per batch the target view's mean
        squared error, with an optimiser step every `grad_batches` batches."""
        self.model.train()
        grad_batches = self.cfg["grad_batches"]
        mse_sum = 0.0
        n_observations = 0
        accumulated = 0
        self.optimiser.zero_grad(set_to_none=True)
        for items in tqdm(self.train_loader, ncols=80, desc=f"Epoch {self.epoch}", leave=False):
            if not items:
                continue
            out = self.model.target_loss(items, self.target_view, amp=self.amp)
            if out["count"] == 0:
                continue
            if not torch.isfinite(out["loss"]):
                print(f"Non-finite loss for {[item['subject'] for item in items]}; skipping the batch")
                continue
            (out["loss"] / grad_batches).backward()
            mse_sum += out["sum"].item()
            n_observations += out["count"]
            del out

            accumulated += 1
            if accumulated == grad_batches:
                self._optimiser_step(self.optimiser)
                self.n_steps += 1
                accumulated = 0
        if accumulated:
            self._optimiser_step(self.optimiser)
            self.n_steps += 1
        return {"train/target_mse": mse_sum / max(n_observations, 1)}

    @torch.no_grad()
    def validate(self) -> tuple[dict, list | None]:
        r"""`val/target_mse`, the target view's mean squared error per
        validation observation, and the first validation batch for images."""
        self.model.eval()
        total = 0.0
        count = 0
        first = None
        for items in self.val_loader:
            if not items:
                continue
            if first is None:
                first = items
            out = self.model.target_loss(items, self.target_view, amp=self.amp)
            total += float(out["sum"])
            count += out["count"]
        return {"val/target_mse": total / count if count else math.nan}, first

    @torch.no_grad()
    def validation_images(self, items: list) -> dict:
        r"""Central slices of the encoder input and the prediction, for the
        first observation of `items`."""
        self.model.eval()
        items = self.model.input_items(items[:1])
        inp = self.model.encoder_input(*self.model.prepare(items))
        volumes = {
            "input/backprojection": inp[0, 0],
            "input/coverage": inp[0, 1],
            "supervised/prediction": self.model.direct_predict(inp, amp=self.amp)[0, 0],
        }
        return volume_images(volumes)

    @property
    def uses_plateaus(self) -> bool:
        return True

    def schedule_log(self) -> dict:
        return {"lr": self.optimiser.param_groups[0]["lr"]}

    def summary(self, log: dict, metrics: dict) -> str:
        return (
            f"Epoch {self.epoch}: train {self.target_view} mse {log.get('train/target_mse', math.nan):.5g}, "
            f"val {self.target_view} mse {metrics['val/target_mse']:.5g}"
        )
