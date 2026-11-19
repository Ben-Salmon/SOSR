r"""Breathhold correction: aligns every short- and long-axis slice of each
subject by matching intensities along their intersections.

    uv run mrisr-breathhold --data_dir <dir> [--max_epochs 1000] [--early_stop_epochs 20]
                            [--lr_sax 2e-3] [--lr_lax 5e-3]

Short-axis slices get whole-voxel in-plane translations; long-axis images a
rigid transform. Writes <subject>/<view>_affine_correction.npy for each view.
Rerun `mrisr-register` afterwards: long-axis corrections only reach the model
through registration_params.pickle.
"""

import argparse
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from .files import find_subjects
from .geometry import batch_normalize_vector, closest_point_on_line, get_image_plane_from_array, plane_intersection
from .interpolation import fast_trilinear_interpolation
from .loss import masked_l2
from .subject import NoValidLongAxisPlanes, SubjectData
from .transforms import lax_voxel_transform

LR_SAX = 2e-3
LR_LAX = 5e-3


def image_center_to_scanner_space(affines: torch.Tensor, shapes: torch.Tensor) -> torch.Tensor:
    centers = torch.zeros((shapes.shape[0], 4), dtype=affines.dtype, device=affines.device)
    centers[:, -1] = 1.0
    centers[:, :2] = shapes[:, :2] / 2
    centers_scanner_space = torch.einsum("ijk,ik->ij", [affines, centers])
    return centers_scanner_space[..., :3]


def find_sampling_center(image_centers1, image_centers2, intersec_lines) -> torch.Tensor:
    centers1_on_line = closest_point_on_line(intersec_lines, image_centers1)
    centers2_on_line = closest_point_on_line(intersec_lines, image_centers2)
    return (centers1_on_line + centers2_on_line) / 2


def compute_intersection_sampling_line(affines, pair_indices, shapes, sampling_step_mm) -> torch.Tensor:
    """A line of points along the intersection of each plane pair, centred on
    the mean of the two image centres projected onto it, sampling_step_mm apart."""
    planes = get_image_plane_from_array(affines)
    planes1, planes2 = planes[pair_indices[:, 0]], planes[pair_indices[:, 1]]
    # We assume the plane pairs always intersect
    intersec_scanner_space = plane_intersection(planes1, planes2)
    num_samples = torch.max(shapes) // 2
    img_centers = image_center_to_scanner_space(affines, shapes)
    sampling_centers = find_sampling_center(
        img_centers[pair_indices[:, 0]], img_centers[pair_indices[:, 1]], intersec_scanner_space
    )

    intersec_dirs = intersec_scanner_space[:, 1] - intersec_scanner_space[:, 0]
    step_vectors = batch_normalize_vector(intersec_dirs)
    step_vectors_ = step_vectors[:, None].tile((1, num_samples, 1))
    dists_from_centers = torch.arange(0, num_samples, dtype=affines.dtype, device=affines.device) - num_samples / 2
    dists_from_centers *= sampling_step_mm
    dists_from_centers_ = dists_from_centers[None, :, None].tile((sampling_centers.shape[0], 1, 3))
    sampling_centers_ = sampling_centers[:, None].tile((1, num_samples, 1))
    sample_lines = sampling_centers_ + step_vectors_ * dists_from_centers_

    assert sample_lines.shape == (pair_indices.shape[0], num_samples, 3)
    return sample_lines


def sample_image_along_scanner_line(
    coords_scanner_space: torch.Tensor,
    images: torch.Tensor,
    affines: torch.Tensor,
    shapes: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = coords_scanner_space.dtype
    device = coords_scanner_space.device
    if coords_scanner_space.shape[-1] == 3:
        last_dim = torch.ones((*coords_scanner_space.shape[:-1], 1), dtype=dtype, device=device)
        coords_scanner_space = torch.cat((coords_scanner_space, last_dim), dim=-1)
    # Scanner space -> voxels, per image and point.
    inverse_affines_ = torch.linalg.inv(affines)[:, None].tile((1, coords_scanner_space.shape[1], 1, 1))
    points_voxel_space = torch.einsum("ijkl,ijl->ijk", [inverse_affines_, coords_scanner_space])
    # The images are 2D: z should be ~0 and is ignored.
    coords_voxel_space = points_voxel_space[:, :, :2]
    # 2D+time images are treated as volumes: trilinear interpolation at each time point.
    coords_voxel_space_t = coords_voxel_space[:, :, None].tile((1, 1, images.shape[-1], 1))
    coords_voxel_space_t = torch.cat((coords_voxel_space_t, t[..., None]), dim=-1)
    sampled_points = fast_trilinear_interpolation(
        images,
        coords_voxel_space_t[..., 0].reshape((images.shape[0], -1)),
        coords_voxel_space_t[..., 1].reshape((images.shape[0], -1)),
        coords_voxel_space_t[..., 2].reshape((images.shape[0], -1)),
    )
    sampled_points = sampled_points.reshape((images.shape[0], coords_scanner_space.shape[1], images.shape[-1]))
    # Which sampled points were inside the image.
    shapes_ = shapes[:, None, :2].tile((1, coords_voxel_space.shape[1], 1))
    out_mask = torch.logical_or(
        (coords_voxel_space < 0.0).any(dim=-1),
        (coords_voxel_space > (shapes_ - 1)).any(dim=-1),
    )
    in_mask = (~out_mask)[..., None].tile((1, 1, images.shape[-1]))
    return sampled_points, in_mask, coords_voxel_space


def sample_intersections(
    images, segs, affines, pair_indices, shapes, spacings
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Both images' intensities along each pair's intersection line, and the
    points the loss compares: (pairs, points, frames) each."""
    sampling_step_size = torch.min(spacings[:, 0])
    sample_coords = compute_intersection_sampling_line(affines, pair_indices, shapes, sampling_step_size)
    i1, i2 = pair_indices[:, 0], pair_indices[:, 1]
    # Time indices (a single frame, so t = 0 everywhere).
    _, _, t = torch.meshgrid(
        *(
            torch.arange(0, n, dtype=sample_coords.dtype, device=sample_coords.device)
            for n in (sample_coords.shape[0], sample_coords.shape[1], images.shape[-1])
        ),
        indexing="ij",
    )
    sampled_im1, sample_mask1, _ = sample_image_along_scanner_line(sample_coords, images[i1], affines[i1], shapes[i1], t)
    sampled_im2, sample_mask2, _ = sample_image_along_scanner_line(sample_coords, images[i2], affines[i2], shapes[i2], t)
    # Heart mask from the segmentations (thresholded, as nearest neighbour).
    sampled_seg1, _, _ = sample_image_along_scanner_line(sample_coords, segs[i1], affines[i1], shapes[i1], t)
    sampled_seg2, _, _ = sample_image_along_scanner_line(sample_coords, segs[i2], affines[i2], shapes[i2], t)
    heart_mask = (sampled_seg1 > 0.5) & (sampled_seg2 > 0.5)
    bounds_mask = sample_mask1 & sample_mask2
    combined_mask = heart_mask & bounds_mask
    # Pairs with no heart point on their intersection use the in-bounds points.
    has_heart = combined_mask.any(dim=2).any(dim=1)
    final_mask = torch.where(has_heart[:, None, None].expand_as(combined_mask), combined_mask, bounds_mask)
    return sampled_im1, sampled_im2, final_mask


def compute_pairwise_loss(images, segs, affines, pair_indices, shapes, spacings) -> torch.Tensor:
    return masked_l2(*sample_intersections(images, segs, affines, pair_indices, shapes, spacings))


@dataclass
class AlignmentResult:
    r"""The outcome of `optimize_affines` for one subject."""

    transformations: list[torch.Tensor]  # What is saved: (2,) per sa slice, (4, 4) per long-axis image.
    voxel_transformations: torch.Tensor  # (n_planes, 4, 4): corrected affines are affines @ these.
    losses: list[float]  # Mean pairwise loss at each step, before that step's update.
    initial_loss: float
    best_epoch: int
    best_loss: float  # Of the saved correction, i.e. after rounding the sa translations.


def optimize_affines(
    subject_data: SubjectData,
    max_epochs: int,
    early_stop_epochs: int,
    lr_sax: float = LR_SAX,
    lr_lax: float = LR_LAX,
    save: bool = True,
) -> AlignmentResult:
    images = subject_data.images_pad
    segs = subject_data.segs_pad
    affines = subject_data.affines
    pair_indices = subject_data.idx_pairs
    shapes = subject_data.shapes
    spacings = subject_data.spacings

    best_loss = compute_pairwise_loss(images, segs, affines, pair_indices, shapes, spacings).mean().item()
    initial_loss = best_loss
    best_epoch = 0
    sa_planes = np.where(np.array(["sa" in n for n in subject_data.names]))[0]
    la_planes = np.where(["sa" not in n for n in subject_data.names])[0]
    sa_affines = affines[sa_planes]
    la_affines = affines[la_planes]
    # Voxel sizes of the long-axis images, so their rotation is rigid in mm.
    la_scales = torch.linalg.norm(la_affines[:, :3, :3], dim=1)
    x_y_voxel_translation = torch.nn.Parameter(
        torch.zeros(size=(sa_affines.shape[0], 2), device=sa_affines.device, dtype=sa_affines.dtype)
    )
    best_x_y_translation = x_y_voxel_translation.clone().detach()
    voxel_translation_rotation = torch.nn.Parameter(
        torch.zeros(size=(la_affines.shape[0], 6), device=la_affines.device, dtype=la_affines.dtype)
    )
    best_translation_rotation = voxel_translation_rotation.clone().detach()

    voxel_transformation = torch.eye(4, device=affines.device, dtype=affines.dtype)
    voxel_transformation = torch.unsqueeze(voxel_transformation, 0).expand_as(affines).contiguous()

    optimizer = torch.optim.Adam(
        [
            {"params": x_y_voxel_translation, "lr": lr_sax},
            {"params": voxel_translation_rotation, "lr": lr_lax},
        ],
    )
    losses = []
    for i in range(max_epochs):
        optimizer.zero_grad()
        epoch_voxel_transformation = voxel_transformation.clone()
        epoch_voxel_transformation[sa_planes, :2, 3] += x_y_voxel_translation
        epoch_voxel_transformation[la_planes] = lax_voxel_transform(voxel_translation_rotation, la_scales)
        epoch_affines = affines @ epoch_voxel_transformation
        loss = compute_pairwise_loss(images, segs, epoch_affines, pair_indices, shapes, spacings).mean()
        losses.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x_y_translation = x_y_voxel_translation.clone().detach()
            best_translation_rotation = voxel_translation_rotation.clone().detach()
            best_epoch = i
        if i > early_stop_epochs and np.min(losses[-early_stop_epochs:]) > best_loss:
            break
        loss.backward()
        optimizer.step()

    best_x_y_translation = torch.round(best_x_y_translation)
    best_translation_rotation = lax_voxel_transform(best_translation_rotation, la_scales)
    best_transformations = [None] * len(affines)
    for i, p in enumerate(sa_planes):
        best_transformations[p] = best_x_y_translation[i]
    for i, p in enumerate(la_planes):
        best_transformations[p] = best_translation_rotation[i]

    best_voxel_transformation = voxel_transformation.clone()
    best_voxel_transformation[sa_planes, :2, 3] += best_x_y_translation
    best_voxel_transformation[la_planes] = best_translation_rotation
    best_affines = affines @ best_voxel_transformation
    best_loss = compute_pairwise_loss(images, segs, best_affines, pair_indices, shapes, spacings).mean().item()
    tqdm.write(f"{subject_data.subject_path.name}: best loss {best_loss:.4g} at epoch {best_epoch}")
    if save:
        subject_data.save_transformations(best_transformations)
    return AlignmentResult(
        transformations=best_transformations,
        voxel_transformations=best_voxel_transformation,
        losses=losses,
        initial_loss=initial_loss,
        best_epoch=best_epoch,
        best_loss=best_loss,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="Directory of subject directories")
    parser.add_argument("--max_epochs", type=int, default=1000, help="Maximum optimisation steps per subject")
    parser.add_argument(
        "--early_stop_epochs", type=int, default=20, help="Stop after this many steps without improvement"
    )
    parser.add_argument("--lr_sax", type=float, default=LR_SAX, help="Adam learning rate of the sa translations")
    parser.add_argument("--lr_lax", type=float, default=LR_LAX, help="Adam learning rate of the long-axis transforms")
    parser.add_argument("--max_subjects", type=int, help="Only the first N subjects (sorted)")
    parser.add_argument("--subjects", nargs="+", help="Only these subject names")
    parser.add_argument(
        "--sax_number",
        type=int,
        help="Correct only this sa stack. For debugging: the default aligns all stacks together, whereas "
        "separate per-stack runs each overwrite the long-axis corrections.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    subjects = find_subjects(args.data_dir, args.sax_number)
    if args.subjects is not None:
        subjects = [s for s in subjects if s.path.name in set(args.subjects)]
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
    skipped = []
    for subject in tqdm(subjects, ncols=80):
        try:
            subject_data = SubjectData(subject, device=args.device)
        except NoValidLongAxisPlanes as error:
            tqdm.write(f"{error}, skipping")
            skipped.append(subject.path.name)
            continue
        optimize_affines(
            subject_data,
            max_epochs=args.max_epochs,
            early_stop_epochs=args.early_stop_epochs,
            lr_sax=args.lr_sax,
            lr_lax=args.lr_lax,
        )
    print(f"Corrected {len(subjects) - len(skipped)} of {len(subjects)} subjects in {args.data_dir}")
    if skipped:
        print(f"Skipped (no valid long-axis plane): {' '.join(skipped)}")
    print("Now rerun mrisr-register (slurm/register.sh): the corrections reach the model through it.")


if __name__ == "__main__":
    main()
