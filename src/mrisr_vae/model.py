r"""The VAE: a Ladder VAE prior over latent volumes, observed through the MRI forward model.

As in Hierarchical DivNoising, the decoder ends in a known measurement model
instead of a learned one. Here that is the forward model of each view:

    z ~ p(z),   x = g(z) on the latent grid,   y_v ~ N(A_v x, sigma_v^2 I).

Maximising the ELBO over the dataset fits p(x) = \int p(x | z) p(z) dz so that
its push-forward through the forward models reproduces the observations: the
empirical-Bayes prior, learned in one training run. The encoder q(z | y) sees
the views through their backprojection onto the latent grid, so it takes any
combination of views at any resolution.

As in COSDD, an optional direct predictor -- a UNet on the same input -- is
trained alongside by regression onto the VAE's (detached) posterior samples.
Under a squared error it learns their mean, E_q[x | y], the MMSE estimate,
in a single pass instead of an average over many samples.

`SupervisedPredictor` is the supervised baseline: the same UNet on the same
input, trained instead against a view the encoder never sees, through that
view's forward model.
"""

import math

import torch
from torch import Tensor, nn

from .backprojection import backproject
from .forward_model import BatchOperator
from .geometry import LatentGeometry
from .likelihood import GaussianLikelihood
from .networks.lvae import LadderVAE
from .networks.unet import UNet


def auto_downsampling(img_size, n_layers: int) -> list[list[int]]:
    r"""Per layer, bottom to top, the halvings along each axis: as in COSDD's
    `get_models`, with the latent grid's size in place of the crop size.

    Each axis is halved floor(log2(size)) - 1 times, if there are enough
    layers. Layers that do not halve an axis are placed at even indices
    first (0, 2, 4, ...), then odd ones, so the bottom latent stays at full
    resolution and the extra layers are spread through the hierarchy.
    """
    n_halvings = [math.floor(math.log2(s)) - 1 for s in img_size]
    difference = [max(n_layers - n, 0) for n in n_halvings]
    downsampling = [[1 for _ in img_size] for _ in range(n_layers)]
    for i in range(len(img_size)):
        j = 0
        while difference[i] > 0:
            # max(..., 1): COSDD's range(n_layers // 2) never ends for one layer
            for k in range(max(n_layers // 2, 1)):
                downsampling[j + k * 2][i] = 0
                difference[i] -= 1
                if difference[i] == 0:
                    break
            j += 1
    return downsampling


def resolve_downsampling(downsampling, img_size, n_layers: int) -> list[list[int]]:
    r"""Per layer, bottom to top, the halvings along each axis, from `auto`
    or a per-layer 0/1 or [d, h, w]."""
    if downsampling == "auto":
        downsampling = auto_downsampling(img_size, n_layers)
    downsampling = [[int(d)] * 3 if isinstance(d, int) else [int(s) for s in d] for d in downsampling]
    if len(downsampling) != n_layers:
        raise ValueError(f"downsampling has {len(downsampling)} entries for {n_layers} layers")
    return downsampling


class ObservationModel(nn.Module):
    r"""What the VAE and the supervised baseline share: the forward operators
    of a batch, the encoder input built from its views, and a direct
    predictor `direct` of the latent volume from that input.

    Subclasses set `geometry`, `lanczos_a`, `coverage_threshold`,
    `likelihood` (for `input_variance`) and `direct` (a UNet, or None).
    """

    # Whether `posterior` draws samples; predictions of `direct` are saved
    # and scored under `direct_name`.
    can_sample = True
    direct_name = "direct"

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def operator(self, items: list[dict]) -> BatchOperator:
        r"""The forward operators of the views of `items`, on the model's device."""
        operators = [{view: d["op"] for view, d in item["views"].items()} for item in items]
        return BatchOperator(operators, self.geometry.size_lvox, lanczos_a=self.lanczos_a).to(self.device)

    def prepare(self, items: list[dict]):
        r"""Moves a batch to the device: its operator, observations and noise levels.

        Returns A and, per observation and view (in `A.views` order), y and the
        measured and extra noise standard deviations as 0-d tensors.
        """
        device = self.device
        A = self.operator(items)
        ys, sigma_meas, sigma_extra = [], [], []
        for item in items:
            views = item["views"].values()
            ys.append([d["y"].to(device, non_blocking=True) for d in views])
            sigma_meas.append([torch.as_tensor(d["sigma_meas"], dtype=torch.float32, device=device) for d in views])
            sigma_extra.append([torch.as_tensor(d["sigma_extra"], dtype=torch.float32, device=device) for d in views])
        return A, ys, sigma_meas, sigma_extra

    @staticmethod
    def input_items(items: list[dict]) -> list[dict]:
        r"""`items` with only the views the encoder is shown: those not marked
        `input: False` by view dropout (or, for the baseline, as its target)."""
        return [
            {**item, "views": {view: d for view, d in item["views"].items() if d.get("input", True)}}
            for item in items
        ]

    def encoder_input(self, A, ys, sigma_meas, sigma_extra) -> Tensor:
        r"""(B, 2, D, H, W): backprojection and coverage. The views are weighted
        by their fixed noise level, not the learned one, so the input does not
        drift during training (see `GaussianLikelihood.input_variance`)."""
        variances = [
            [self.likelihood.input_variance(view, m, e) for view, m, e in zip(views, ms, es, strict=True)]
            for views, ms, es in zip(A.views, sigma_meas, sigma_extra, strict=True)
        ]
        with torch.no_grad():
            return backproject(A, ys, variances, coverage_threshold=self.coverage_threshold)

    def direct_predict(self, inp: Tensor, amp: bool = True) -> Tensor:
        r"""The direct predictor's estimate from the encoder input: (B, 1, D, H, W) in float32."""
        if self.direct is None:
            raise RuntimeError("This model was built without a direct predictor")
        device_type = inp.device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp and device_type == "cuda"):
            return self.direct(inp).float()

    @torch.no_grad()
    def predict_direct(self, items: list[dict], amp: bool = True) -> Tensor:
        r"""The direct predictor's estimate given every view of `items`, in one pass: (B, 1, D, H, W)."""
        A, ys, sigma_meas, sigma_extra = self.prepare(items)
        return self.direct_predict(self.encoder_input(A, ys, sigma_meas, sigma_extra), amp=amp)


class MRIVAE(ObservationModel):
    def __init__(
        self,
        geometry: LatentGeometry,
        n_layers: int = 8,
        z_dim: int = 32,
        n_filters: int = 32,
        blocks_per_layer: int = 1,
        downsampling="auto",
        scale_initialisation: bool = False,
        checkpointed: bool = True,
        learn_noise: bool = True,
        min_sigma: float = 1e-3,
        noise: str = "per_group",
        init_sigma: float = 0.05,
        relative_std: dict | None = None,
        lanczos_a: int = 3,
        coverage_threshold: float = 1e-2,
        direct_predictor: bool = True,
    ):
        r"""
        Arguments:
            downsampling: `auto` (see `auto_downsampling`), or per layer,
                bottom to top, whether it halves the resolution: 0/1, or
                [d, h, w] per axis.
            direct_predictor: Also train a UNet, of the same depth and width as
                the Ladder VAE, to predict E[x | y] in one pass.
            noise, init_sigma, relative_std: The likelihood's noise model (see
                `GaussianLikelihood`).
        """
        super().__init__()
        self.geometry = geometry
        self.lanczos_a = lanczos_a
        self.coverage_threshold = coverage_threshold
        downsampling = resolve_downsampling(downsampling, geometry.size_lvox, n_layers)
        self.downsampling = downsampling
        self.lvae = LadderVAE(
            colour_channels=2,
            img_size=geometry.size_lvox,
            s_code_channels=1,
            z_dims=[z_dim] * n_layers,
            blocks_per_layer=blocks_per_layer,
            n_filters=n_filters,
            scale_initialisation=scale_initialisation,
            downsampling=downsampling,
            checkpointed=checkpointed,
            dimensions=3,
        )
        self.likelihood = GaussianLikelihood(
            geometry.views,
            learn=learn_noise,
            min_sigma=min_sigma,
            noise=noise,
            init_sigma=init_sigma,
            relative_std=relative_std,
        )
        self.direct = None
        if direct_predictor:
            self.direct = UNet(
                in_channels=2,
                out_channels=1,
                blocks_per_layer=blocks_per_layer,
                n_filters=n_filters,
                n_layers=n_layers,
                downsampling=downsampling,
                scale_initialisation=scale_initialisation,
                checkpointed=checkpointed,
                dimensions=3,
            )

    def decode(self, inp: Tensor, amp: bool = True):
        r"""Runs the Ladder VAE, sampling every latent from q. Returns the latent
        volume x (B, 1, D, H, W) in float32 and the LVAE's output (for the KL)."""
        device_type = inp.device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp and device_type == "cuda"):
            out = self.lvae(inp)
        return out["s_code"].float(), out

    def elbo(self, items: list[dict], amp: bool = True, free_bits: float = 0.0, kl_weight: float = 1.0) -> dict:
        r"""The negative ELBO of a batch, in nats per observed voxel.

        The encoder sees only the views not marked `input: False`, but every
        view is in the likelihood. q(z | some views) still bounds log p(all
        views), so this is still an ELBO; it makes the latents predict the
        views they were not shown.

        `loss` is what is minimised: as `nelbo`, but with the KL clamped below
        at `free_bits` (per observed voxel), as in COSDD, to keep the latents
        in use early in training, then scaled by `kl_weight` (the KL warm-up).

        The direct predictor is trained separately, by `direct_loss` on the
        returned `input` and detached `x`, after the VAE's backward pass (see
        `Trainer.train_epoch`).
        """
        A, ys, sigma_meas, sigma_extra = self.prepare(items)
        inputs = self.input_items(items)
        if all(len(a["views"]) == len(b["views"]) for a, b in zip(inputs, items, strict=True)):
            inp = self.encoder_input(A, ys, sigma_meas, sigma_extra)
        else:
            inp = self.encoder_input(*self.prepare(inputs))
        x, out = self.decode(inp, amp=amp)
        # The forward model runs in float32, outside autocast: in bfloat16 the
        # rounding of y_hat alone is ~0.4% of the signal.
        with torch.autocast(device_type=x.device.type, enabled=False):
            y_hat = A(x)

        nll = x.new_zeros(())
        n_rows = x.new_zeros(())
        for b, views in enumerate(A.views):
            for v, view in enumerate(views):
                sigma = self.likelihood.sigma(view, sigma_meas[b][v], sigma_extra[b][v])
                term, n = self.likelihood.nll(ys[b][v], y_hat[b][v], A.op(b, view).row_mask, sigma)
                nll = nll + term
                n_rows = n_rows + n

        kl_layers = self.lvae.kl_per_layer(out["q_list"], out["p_list"])
        kl = kl_layers.sum()
        n = n_rows.clamp_min(1)
        kl_term = kl / n
        if free_bits > 0:
            kl_term = torch.clamp(kl_term, min=free_bits)
        return {
            "loss": nll / n + kl_weight * kl_term,
            "nelbo": (nll + kl) / n,
            "nll": nll / n,
            "kl": kl / n,
            "kl_layers": kl_layers / n,
            "n_rows": n_rows,
            "x": x,
            "input": inp,
        }

    @torch.no_grad()
    def direct_targets(self, items: list[dict], amp: bool = True) -> tuple[Tensor, Tensor]:
        r"""The direct predictor's training pair when the VAE is frozen: the
        encoder input of the views not marked `input: False`, and one posterior
        sample given it, as `elbo` returns them, without the likelihood or KL."""
        A, ys, sigma_meas, sigma_extra = self.prepare(self.input_items(items))
        inp = self.encoder_input(A, ys, sigma_meas, sigma_extra)
        x, _ = self.decode(inp, amp=amp)
        return inp, x

    def direct_loss(self, inp: Tensor, target: Tensor, amp: bool = True) -> Tensor:
        r"""The direct predictor's mean squared error against posterior samples
        `target` (detached, so only the predictor learns from it). Its minimiser
        is their mean, E_q[x | y]."""
        return torch.mean((self.direct_predict(inp, amp=amp) - target.detach()) ** 2)

    @torch.no_grad()
    def heldout_error(self, items: list[dict], amp: bool = True, direct: bool = False) -> tuple[Tensor, int]:
        r"""Squared error on a held-out view, summed over observations.

        For each observation with two or more views, the last view in VIEWS
        order is held out, one posterior sample is drawn given the rest (or,
        with `direct`, the direct predictor's estimate is taken), and the
        result is pushed through the held-out view's forward model. The same
        metric as diffusion-priors' `calculate_heldout_error`, which also uses
        one posterior sample: it rewards a prior for predicting what it was not
        shown. Fix the random seed to compare values.

        Returns the sum of the per-observation mean squared errors over the
        supported rows, and the number of observations.
        """
        conditioning, heldout = [], []
        for item in items:
            views = [view for view in self.geometry.views if view in item["views"]]
            if len(views) < 2:
                continue
            conditioning.append({**item, "views": {view: item["views"][view] for view in views[:-1]}})
            heldout.append({**item, "views": {views[-1]: item["views"][views[-1]]}})
        total = torch.zeros((), device=self.device)
        if not conditioning:
            return total, 0
        A, ys, sigma_meas, sigma_extra = self.prepare(conditioning)
        inp = self.encoder_input(A, ys, sigma_meas, sigma_extra)
        if direct:
            x = self.direct_predict(inp, amp=amp)
        else:
            x, _ = self.decode(inp, amp=amp)
        A_heldout, ys_heldout, _, _ = self.prepare(heldout)
        with torch.autocast(device_type=x.device.type, enabled=False):
            y_hat = A_heldout(x)
        count = 0
        for b, views in enumerate(A_heldout.views):
            (view,) = views
            mask = A_heldout.op(b, view).row_mask
            n = mask.sum()
            if n > 0:
                total = total + torch.sum(mask * (ys_heldout[b][0] - y_hat[b][0]) ** 2) / n
                count += 1
        return total, count

    @torch.no_grad()
    def posterior(self, items: list[dict], n_samples: int = 1, amp: bool = True) -> Tensor:
        r"""Latent volumes x = g(z), z ~ q(z | y): (n_samples, B, 1, D, H, W)."""
        A, ys, sigma_meas, sigma_extra = self.prepare(items)
        inp = self.encoder_input(A, ys, sigma_meas, sigma_extra)
        return torch.stack([self.decode(inp, amp=amp)[0] for _ in range(n_samples)])

    @torch.no_grad()
    def sample_prior(self, n: int, amp: bool = True) -> Tensor:
        r"""Latent volumes from the prior: (n, 1, D, H, W)."""
        device_type = self.device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp and device_type == "cuda"):
            x = self.lvae.sample_from_prior(n)
        return x.float()


class SupervisedPredictor(ObservationModel):
    r"""The supervised baseline: the VAE's direct predictor, a UNet on the
    same encoder input, trained to predict a view the encoder is never shown.

    The UNet predicts a volume x on the latent grid; the target view's forward
    model takes it to that view's voxels, where it is compared with the view
    by mean squared error. No prior and no posterior: one estimate per input.

    Arguments are the UNet's, as `MRIVAE` builds its direct predictor, and the
    likelihood's noise settings, which only weight the views in the encoder
    input (`GaussianLikelihood.input_variance`), so the input is the VAE's.
    """

    can_sample = False
    direct_name = "supervised"

    def __init__(
        self,
        geometry: LatentGeometry,
        n_layers: int = 8,
        n_filters: int = 32,
        blocks_per_layer: int = 1,
        downsampling="auto",
        scale_initialisation: bool = False,
        checkpointed: bool = True,
        min_sigma: float = 1e-3,
        noise: str = "per_group",
        init_sigma: float = 0.05,
        relative_std: dict | None = None,
        lanczos_a: int = 3,
        coverage_threshold: float = 1e-2,
    ):
        super().__init__()
        self.geometry = geometry
        self.lanczos_a = lanczos_a
        self.coverage_threshold = coverage_threshold
        self.downsampling = resolve_downsampling(downsampling, geometry.size_lvox, n_layers)
        self.likelihood = GaussianLikelihood(
            geometry.views,
            learn=False,
            min_sigma=min_sigma,
            noise=noise,
            init_sigma=init_sigma,
            relative_std=relative_std,
        )
        self.direct = UNet(
            in_channels=2,
            out_channels=1,
            blocks_per_layer=blocks_per_layer,
            n_filters=n_filters,
            n_layers=n_layers,
            downsampling=self.downsampling,
            scale_initialisation=scale_initialisation,
            checkpointed=checkpointed,
            dimensions=3,
        )

    def target_loss(self, items: list[dict], target_view: str, amp: bool = True) -> dict:
        r"""Mean squared error of the prediction of `target_view`.

        The encoder sees the views not marked `input: False` (the target view
        always is). The prediction x is pushed through the target view's
        forward model, in float32 outside autocast as in `MRIVAE.elbo`, and
        compared with it on the rows that forward model supports: per
        observation the mean over those rows, as in `MRIVAE.heldout_error`.

        Returns `loss`, the mean over observations with supported rows; `sum`
        and `count`, their sum and number; and the encoder `input` and `x`.
        """
        missing = [item["subject"] for item in items if target_view not in item["views"]]
        if missing:
            raise ValueError(f"No {target_view} for {missing}: load the data with target_views=[{target_view!r}]")
        inp = self.encoder_input(*self.prepare(self.input_items(items)))
        x = self.direct_predict(inp, amp=amp)
        targets = [{**item, "views": {target_view: item["views"][target_view]}} for item in items]
        A, ys, _, _ = self.prepare(targets)
        with torch.autocast(device_type=x.device.type, enabled=False):
            y_hat = A(x)
        total = x.new_zeros(())
        count = 0
        for b in range(len(targets)):
            mask = A.op(b, target_view).row_mask
            n = mask.sum()
            if n > 0:
                total = total + torch.sum(mask * (ys[b][0] - y_hat[b][0]) ** 2) / n
                count += 1
        return {"loss": total / max(count, 1), "sum": total, "count": count, "input": inp, "x": x}


def load_pretrained_vae(model: MRIVAE, path: str):
    r"""Loads the Ladder VAE and likelihood of the checkpoint at `path` into
    `model`, but not its direct predictor, which keeps its fresh weights."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["geometry_config"] != model.geometry.config:
        raise ValueError(f"{path} was trained on a different latent geometry")
    weights = {key: value for key, value in state["model"].items() if not key.startswith("direct.")}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    missing = [key for key in missing if not key.startswith("direct.")]
    if missing or unexpected:
        raise ValueError(f"{path} does not match the model: missing {missing}, unexpected {unexpected}")


MODEL_TYPES = ("vae", "supervised")


def build_model(config: dict, geometry: LatentGeometry) -> MRIVAE | SupervisedPredictor:
    r"""The model described by a training config (see `train.DEFAULTS`):
    `model: type` vae (`MRIVAE`) or supervised (`SupervisedPredictor`)."""
    model_cfg = dict(config["model"])
    # Checkpoints from before the supervised baseline lack `type`.
    model_type = model_cfg.pop("type", "vae")
    likelihood = config["likelihood"]
    noise = {
        "min_sigma": likelihood["min_sigma"],
        # Checkpoints from before the global noise model lack these keys.
        "noise": likelihood.get("noise", "per_group"),
        "init_sigma": likelihood.get("init_sigma", 0.05),
        "relative_std": likelihood.get("relative_std"),
    }
    if model_type == "supervised":
        # The VAE's options, which the UNet alone does not use.
        for key in ("z_dim", "direct_predictor"):
            model_cfg.pop(key, None)
        return SupervisedPredictor(geometry, **noise, **model_cfg)
    if model_type != "vae":
        raise ValueError(f"`model: type` must be one of {', '.join(MODEL_TYPES)}, not {model_type!r}")
    return MRIVAE(geometry, learn_noise=likelihood["learn_noise"], **noise, **model_cfg)
