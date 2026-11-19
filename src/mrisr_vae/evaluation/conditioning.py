r"""Which views a prediction is conditioned on, and which voxels are scored.

Conditioning sets: `all` (every view the model was trained on), a
comma-separated list of views or view groups, e.g. `sa` (every short-axis
stack), `la_2ch` or `sa,la_2ch`, or a group with `_first`, e.g. `sa_first`:
each subject's lowest-numbered short-axis stack alone, the stack the SMORE and
interpolation baselines are given. Each set is normalised from its own views
alone, so nothing else leaks into the input.

Region: `<group>_first` (e.g. `sa_first`) scores only the held-out voxels
inside the subject's lowest-numbered stack of that group -- within its
in-plane field of view, and from half a slice thickness below its first slice
to half a thickness above its last, gaps between slices included. Elsewhere,
e.g. the atria above a short-axis stack, no training view has data. The region
depends only on which stacks the subject has and their geometry, so it is the
same for every method and conditioning set. None scores wherever the held-out
view's forward model is supported.
"""

import functools
from collections.abc import Callable
from typing import NamedTuple

import torch
from tqdm import tqdm

from ..data import NORMALISED_STD
from ..forward_model import BatchOperator, apply_affine, invert_affine
from ..geometry import LatentGeometry
from ..likelihood import view_group

FIRST_SUFFIX = "_first"


def first_group(spec: str) -> str | None:
    r"""The group of a `<group>_first` conditioning set or region, else None."""
    return spec[: -len(FIRST_SUFFIX)] if spec.endswith(FIRST_SUFFIX) else None


def view_index(view: str):
    r"""Sort key: a view's trailing number (sa_10 after sa_2), -1 if none."""
    suffix = view.rsplit("_", 1)[-1]
    return (int(suffix) if suffix.isdigit() else -1, view)


def resolve_conditioning(spec: str, views: list[str]) -> list[str]:
    r"""The views of a conditioning set: `all`, or comma-separated views or
    groups. For `<group>_first`, the group's views, of which each subject is
    given its lowest-numbered one (`norm_views_for`, `item_conditioning`)."""
    if spec == "all":
        return list(views)
    group = first_group(spec)
    if group is not None:
        candidates = [view for view in views if view_group(view) == group]
        if not candidates:
            raise ValueError(f"`{group}` is no group of the views the model was trained on: {views}")
        return candidates
    names = [name for name in spec.split(",") if name]
    for name in names:
        if not any(view == name or view_group(view) == name for view in views):
            hint = f" (`{name}` must be a conditioning set on its own)" if first_group(name) else ""
            raise ValueError(f"`{name}` matches none of the views the model was trained on: {views}{hint}")
    return [view for view in views if any(view == name or view_group(view) == name for name in names)]


def lowest_numbered(candidates, available) -> list[str]:
    r"""The lowest-numbered of `candidates` among `available`, as a list (empty if none)."""
    present = [view for view in available if view in candidates]
    return [min(present, key=view_index)] if present else []


def norm_views_for(spec: str, conditioning_views: list[str]):
    r"""The dataset's `norm_views` for a conditioning set: its views, or for
    `<group>_first` a (picklable) function picking each subject's
    lowest-numbered stack of them, which alone then sets the normalisation."""
    if first_group(spec) is not None:
        return functools.partial(lowest_numbered, tuple(conditioning_views))
    return conditioning_views


def item_conditioning(spec: str, item: dict, conditioning_views: list[str]) -> list[str]:
    r"""The views `item` is conditioned on: for `<group>_first`, the stack the
    dataset picked (and normalised from); otherwise the set's views."""
    if first_group(spec) is not None:
        return list(item["norm_views"])
    return conditioning_views


def conditioning_label(spec: str) -> str:
    r"""A conditioning set's name in file paths and tables."""
    return spec.replace(",", "+")


def check_region(region: str | None, views: list[str]):
    r"""Raises ValueError unless `region` is None or `<group>_first` of a group of `views`."""
    if region is None:
        return
    group = first_group(region)
    if group is None:
        raise ValueError(f"region must be `<group>_first` (e.g. sa_first) or null, not `{region}`")
    if not any(view_group(view) == group for view in views):
        raise ValueError(f"region `{region}`: `{group}` is no group of the views the model was trained on: {views}")


def region_stack(item: dict, region: str, views) -> str | None:
    r"""The stack bounding the scored region: the item's lowest-numbered stack
    of the `<group>_first` region's group among `views`. None if it has none."""
    group = first_group(region)
    candidates = [view for view in item["views"] if view in views and view_group(view) == group]
    return min(candidates, key=view_index) if candidates else None


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
    (`<group>_first`), only those inside the stack `region_stack` picks from
    the item's stacks in the geometry's views (`stack_region`). None if the
    item lacks the view or no voxel is left.
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
