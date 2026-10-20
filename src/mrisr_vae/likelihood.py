r"""Gaussian likelihood of the observed views given the latent volume."""

import math

import torch
from torch import Tensor, nn

LOG_2PI = math.log(2 * math.pi)
NOISE_MODES = ("per_group", "global")


def view_group(view: str) -> str:
    r"""The group whose noise scale a view shares: its name up to the first
    underscore, e.g. `sa` for `sa_3` and `la` for `la_2ch`."""
    return view.split("_")[0]


class GaussianLikelihood(nn.Module):
    r"""p(y_v | x) = N(y_v; A_v x, sigma_v^2 I) on the rows the forward model supports.

    With `noise="per_group"` the noise standard deviation of a view is

        sigma_v^2 = sigma_meas^2 + (s_g * sigma_extra)^2 + min_sigma^2,

    where `sigma_meas` is the measured noise, `sigma_extra` the per-view extra
    noise of the config (`extra_noise_std`, there to absorb misalignment left
    by the breathhold correction), and s_g = exp(log_scale[g]) a learned scale
    shared by the views of group g. With `learn=False` the scales stay at 1,
    which is the diffusion-priors weighting. A group with no extra noise has
    nothing to scale.

    With `noise="global"` it is

        sigma_v^2 = sigma_meas^2 + (r_v * sigma)^2 + min_sigma^2,

    where sigma = exp(log_sigma), in normalised units, is a single scale shared
    by every view, starting at `init_sigma`, and r_v a fixed ratio for view v
    (`relative_std`, 1 for views not given). `sigma_extra` is not used. At its
    optimum sigma^2 is the mean of residual^2 / r_v^2 over all observed rows,
    so it tracks the model's misfit while r_v sets how much less a view is
    trusted: its residuals count 1 / r_v^2 as much.
    """

    def __init__(
        self,
        views,
        learn: bool = True,
        min_sigma: float = 1e-3,
        noise: str = "per_group",
        init_sigma: float = 0.05,
        relative_std: dict | None = None,
    ):
        super().__init__()
        if noise not in NOISE_MODES:
            raise ValueError(f"noise must be one of {NOISE_MODES}, not {noise!r}")
        self.noise = noise
        self.groups = sorted({view_group(view) for view in views})
        self.min_sigma = min_sigma
        if noise == "per_group":
            self.log_scale = nn.Parameter(torch.zeros(len(self.groups)), requires_grad=learn)
        else:
            if not init_sigma > 0:
                raise ValueError("init_sigma must be positive")
            relative_std = dict(relative_std or {})
            for view, ratio in relative_std.items():
                if not ratio > 0:
                    raise ValueError(f"relative_std of `{view}` must be positive")
            # Held-out views are not in `views`, so a ratio may name a view
            # that training never sees; it is kept for it all the same.
            self.relative_std = relative_std
            self.init_sigma = init_sigma
            self.log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma)), requires_grad=learn)

    def scale(self, view: str) -> Tensor:
        return torch.exp(self.log_scale[self.groups.index(view_group(view))])

    def ratio(self, view: str) -> float:
        r"""`global` only: the fixed ratio r_v of the view."""
        return float(self.relative_std.get(view, 1.0))

    def sigma(self, view: str, sigma_meas: Tensor, sigma_extra: Tensor) -> Tensor:
        if self.noise == "per_group":
            modelled = self.scale(view) * sigma_extra
        else:
            modelled = self.ratio(view) * torch.exp(self.log_sigma)
        variance = sigma_meas**2 + modelled**2
        return torch.sqrt(variance + self.min_sigma**2)

    def input_variance(self, view: str, sigma_meas: Tensor, sigma_extra: Tensor) -> Tensor:
        r"""The variance that weights the view in the encoder's input. Fixed,
        not learned, so the input does not drift during training: the config's
        noise, or for `global` the ratio times `init_sigma`."""
        if self.noise == "per_group":
            return sigma_meas**2 + sigma_extra**2
        return sigma_meas**2 + (self.ratio(view) * self.init_sigma) ** 2

    def log_values(self) -> dict:
        r"""The learned noise, for logging."""
        if self.noise == "per_group":
            return {f"noise_scale/{g}": math.exp(s) for g, s in zip(self.groups, self.log_scale.tolist())}
        # A view's sigma is this times its ratio.
        return {"noise_sigma": math.exp(self.log_sigma.item())}

    @staticmethod
    def nll(y: Tensor, y_hat: Tensor, row_mask: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor]:
        r"""Negative log likelihood summed over the rows in `row_mask`, and their number."""
        residual = (y - y_hat) * row_mask
        n_rows = row_mask.sum()
        nll = 0.5 * torch.sum(residual**2) / sigma**2 + n_rows * (torch.log(sigma) + 0.5 * LOG_2PI)
        return nll, n_rows
