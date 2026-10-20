r"""The latent (high-resolution) grid.

Computes what `forward_model.init_geometry` computes in diffusion-priors, but
as an object that is passed around rather than module globals. Globals set in
the main process do not reach DataLoader workers started with spawn or
forkserver, and the workers build the forward operators.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LatentGeometry:
    config: dict  # the geometry yaml, kept for checkpoints
    views: list[str]
    su_per_lvox: Tensor  # 0-d: the grid is isotropic
    size_lvox: tuple[int, int, int]
    min_coords_su: Tensor  # (3,)
    max_coords_su: Tensor  # (3,)
    coords_su: Tensor  # (3, D, H, W), lvox centres in su
    lvox_to_su_affine: Tensor  # (4, 4)

    @classmethod
    def from_config(cls, config: dict) -> "LatentGeometry":
        scale_factor = config["SCALE_FACTOR"]
        su_per_lvox = torch.tensor(config["RAW_SU_PER_LVOX"]) / scale_factor
        if su_per_lvox.ndim != 0:
            raise ValueError("RAW_SU_PER_LVOX must be a single number: the grid is isotropic")
        x_size_su = torch.tensor(config["x_size_su"])
        size_lvox = torch.round(x_size_su / su_per_lvox)
        size_su = size_lvox * su_per_lvox
        min_coords_su = torch.tensor(config["X_MIN_COORDS_SU"], dtype=torch.float32)
        max_coords_su = min_coords_su + size_su

        # Cell-centre grid: lvox i sits at min + (i + 0.5) * su_per_lvox.
        axes = [
            min_coords_su[i]
            + (torch.arange(int(size_lvox[i]), dtype=torch.float32) + 0.5) * su_per_lvox
            for i in range(3)
        ]
        coords_su = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)

        lvox_to_su_affine = torch.eye(4)
        lvox_to_su_affine[:3, :3] *= su_per_lvox
        lvox_to_su_affine[:3, 3] = min_coords_su + 0.5 * su_per_lvox

        return cls(
            config=dict(config),
            views=list(config["VIEWS"]),
            su_per_lvox=su_per_lvox,
            size_lvox=tuple(int(s) for s in size_lvox),
            min_coords_su=min_coords_su,
            max_coords_su=max_coords_su.to(torch.float32),
            coords_su=coords_su,
            lvox_to_su_affine=lvox_to_su_affine,
        )
