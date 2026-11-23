r"""Breathhold correction: aligns every short- and long-axis slice of each
subject by matching intensities along their intersections.

    uv run mrisr-breathhold --data_dir <dir> [--rounds 3] [--search_radius 30]
                            [--lax_steps 300] [--lax_patience 30] [--lr_lax 1e-2]
                            [--seg_weight 1.0] [--metric ncc] [--line_mm 120]

Short-axis slices get whole-voxel in-plane translations; long-axis images a
rigid transform. The two are found in turn: an exhaustive search over every
short-axis slice's translation (exact, as the slices are parallel and so only
paired with long-axis planes, which it holds fixed), then gradient descent on
the long-axis transforms, repeated until the search stops changing anything.
The search does not need a starting point near the answer, which gradient
descent on the translations did.

Intensities are compared by normalised cross-correlation, along a 120 mm
stretch of each intersection centred between the two images' centres: slices
of a stack converted from separate series can differ in gain and contrast,
and the chest wall and abdomen further out move differently from the heart
between breath-holds. Either pulled plain L2 over the whole line 30 mm off on
a slice that differed in contrast from the long-axis images.

Segmentations are optional. Where both planes of a pair are segmented, their
hearts must overlap along the intersection as well as their intensities match.

Writes <subject>/<view>_affine_correction.npy for each view. Rerun
`mrisr-register` afterwards: long-axis corrections only reach the model through
registration_params.pickle.
"""

import argparse
from dataclasses import dataclass, replace

import torch
from tqdm import tqdm

from .files import find_subjects
from .geometry import batch_normalize_vector, closest_point_on_line, get_image_plane_from_array, plane_intersection
from .interpolation import fast_trilinear_interpolation
from .loss import METRICS, intersection_loss
from .subject import NoValidLongAxisPlanes, SubjectData
from .transforms import lax_voxel_transform

ROUNDS = 3  # Short-axis searches, each followed by a long-axis refinement.
SEARCH_RADIUS = 30  # Voxels either side of the acquired position, in x and y.
FINE_RADIUS = 2  # Of the intensity-only search around the heart-guided one.
LAX_STEPS = 300  # Maximum Adam steps per long-axis refinement.
LAX_PATIENCE = 30  # Stop a refinement after this many steps without improvement.
LR_LAX = 1e-2  # Radians and voxels.
SEG_WEIGHT = 1.0  # Of the heart-overlap term; intensities are scaled to about [0, 1].
METRIC = "ncc"  # Intensity dissimilarity: "ncc" or "l2" (see `intersection_loss`).
LINE_MM = 120.0  # Length of each intersection compared; None for the original max(shape) / 2 points.


@dataclass(frozen=True)
class LossOptions:
    seg_weight: float = SEG_WEIGHT
    metric: str = METRIC
    line_mm: float | None = LINE_MM


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


def compute_intersection_sampling_line(affines, pair_indices, shapes, sampling_step_mm, line_mm=None) -> torch.Tensor:
    """A line of points along the intersection of each plane pair, centred on
    the mean of the two image centres projected onto it, sampling_step_mm apart:
    line_mm long, or max(shapes) / 2 points if line_mm is None."""
    planes = get_image_plane_from_array(affines)
    planes1, planes2 = planes[pair_indices[:, 0]], planes[pair_indices[:, 1]]
    # We assume the plane pairs always intersect
    intersec_scanner_space = plane_intersection(planes1, planes2)
    if line_mm is None:
        num_samples = int(torch.max(shapes)) // 2
    else:
        num_samples = max(int(round(line_mm / float(sampling_step_mm))), 1)
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


def sample_intersections(images, segs, has_seg, affines, pair_indices, shapes, spacings, line_mm=LINE_MM):
    r"""Along each pair's intersection line: both images' intensities, both
    heart masks (soft, from bilinear sampling of the binary masks), the points
    to compare, all (pairs, points, frames), and whether both planes of the
    pair are segmented, (pairs,).

    The points compared are those inside both images and inside the union of
    the hearts of the pair's segmented planes, or all in-bounds points if
    neither plane is segmented or the hearts miss the line. A union, unlike an
    intersection, does not shrink as the hearts move apart, so misaligned hearts
    are compared heart against background and score badly."""
    sampling_step_size = torch.min(spacings[:, 0])
    sample_coords = compute_intersection_sampling_line(affines, pair_indices, shapes, sampling_step_size, line_mm)
    i1, i2 = pair_indices[:, 0], pair_indices[:, 1]
    # Time indices (a single frame, so t = 0 everywhere).
    _, _, t = torch.meshgrid(
        *(
            torch.arange(0, n, dtype=sample_coords.dtype, device=sample_coords.device)
            for n in (sample_coords.shape[0], sample_coords.shape[1], images.shape[-1])
        ),
        indexing="ij",
    )
    sampled_im1, in1, _ = sample_image_along_scanner_line(sample_coords, images[i1], affines[i1], shapes[i1], t)
    sampled_im2, in2, _ = sample_image_along_scanner_line(sample_coords, images[i2], affines[i2], shapes[i2], t)
    heart1, _, _ = sample_image_along_scanner_line(sample_coords, segs[i1], affines[i1], shapes[i1], t)
    heart2, _, _ = sample_image_along_scanner_line(sample_coords, segs[i2], affines[i2], shapes[i2], t)
    bounds_mask = in1 & in2
    # Unsegmented planes have all-zero masks, so they add nothing to the union.
    heart_mask = bounds_mask & ((heart1 > 0.5) | (heart2 > 0.5))
    use_heart = heart_mask.any(dim=2).any(dim=1)
    mask = torch.where(use_heart[:, None, None], heart_mask, bounds_mask)
    both_segmented = has_seg[i1] & has_seg[i2]
    return sampled_im1, sampled_im2, heart1, heart2, mask, both_segmented


def compute_pairwise_loss(
    images, segs, has_seg, affines, pair_indices, shapes, spacings, options: LossOptions = LossOptions()
) -> torch.Tensor:
    r"""The loss of each pair, (pairs,)."""
    samples = sample_intersections(images, segs, has_seg, affines, pair_indices, shapes, spacings, options.line_mm)
    return intersection_loss(*samples, options.seg_weight, options.metric)


def voxel_transformations(n_planes, sa_planes, la_planes, sax_shifts, lax_params, la_scales) -> torch.Tensor:
    r"""(n_planes, 4, 4) voxel transforms: corrected affines are affines @ these.
    `sax_shifts` (n_sa, 2) are in-plane voxel translations, `lax_params`
    (n_la, 6) rigid long-axis corrections (see `lax_voxel_transform`)."""
    transforms = torch.eye(4, dtype=lax_params.dtype, device=lax_params.device).repeat(n_planes, 1, 1)
    transforms[sa_planes, :2, 3] = sax_shifts
    transforms[la_planes] = lax_voxel_transform(lax_params, la_scales)
    return transforms


def shift_grid(radius: int, dtype, device) -> torch.Tensor:
    r"""Every whole-voxel (x, y) shift within `radius` voxels, ((2r+1)^2, 2)."""
    r = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    return torch.cartesian_prod(r, r)


def search_sax_shifts(
    images, segs, has_seg, affines, sa_planes, sa_la_pairs, pair_slice, shapes, spacings, offsets, options,
    centres=None,
) -> torch.Tensor:
    r"""The best in-plane shift of each short-axis slice among `centres` +
    `offsets`, (n_sa, 2), with the long-axis planes held where `affines` puts
    them.

    Short-axis slices are parallel, so they are only ever paired with long-axis
    planes: with those fixed, each slice's loss depends on its own shift only.
    Each offset is therefore applied to every slice at once, and the best is
    chosen per slice, which is the exact optimum over the offsets.

    sa_la_pairs: the short/long-axis pairs. pair_slice: each one's slice, as an
    index into `sa_planes`. centres: (n_sa, 2), zero by default.
    """
    if centres is None:
        centres = torch.zeros((len(sa_planes), 2), dtype=affines.dtype, device=affines.device)
    scores = torch.zeros((len(offsets), len(sa_planes)), dtype=affines.dtype, device=affines.device)
    eye = torch.eye(4, dtype=affines.dtype, device=affines.device).repeat(len(affines), 1, 1)
    for k, offset in enumerate(offsets):
        transforms = eye.clone()
        transforms[sa_planes, :2, 3] = centres + offset
        loss = compute_pairwise_loss(
            images, segs, has_seg, affines @ transforms, sa_la_pairs, shapes, spacings, options
        )
        scores[k].index_add_(0, pair_slice, loss)
    return centres + offsets[scores.argmin(dim=0)]


def refine_lax(
    images, segs, has_seg, affines, pair_indices, shapes, spacings,
    sa_planes, la_planes, sax_shifts, lax_params, la_scales, steps, patience, lr, options,
) -> torch.Tensor:
    r"""Adam on the long-axis rigid transforms, short-axis shifts fixed, from
    `lax_params`. Returns the best parameters found."""
    params = torch.nn.Parameter(lax_params.clone())
    optimizer = torch.optim.Adam([params], lr=lr)
    best_params, best_loss, best_step = lax_params.clone(), float("inf"), 0
    for step in range(steps):
        optimizer.zero_grad()
        transforms = voxel_transformations(len(affines), sa_planes, la_planes, sax_shifts, params, la_scales)
        loss = compute_pairwise_loss(
            images, segs, has_seg, affines @ transforms, pair_indices, shapes, spacings, options
        ).mean()
        if loss.item() < best_loss:
            best_params, best_loss, best_step = params.detach().clone(), loss.item(), step
        if step - best_step >= patience:
            break
        loss.backward()
        optimizer.step()
    return best_params


@dataclass
class AlignmentResult:
    r"""The outcome of `optimize_affines` for one subject."""

    transformations: list[torch.Tensor]  # What is saved: (2,) per sa slice, (4, 4) per long-axis image.
    voxel_transformations: torch.Tensor  # (n_planes, 4, 4): corrected affines are affines @ these.
    stages: list[str]  # "initial", then "sax search k" and "lax refinement k" in the order run.
    losses: list[float]  # Mean pairwise loss (heart term included) after each stage.
    rounds: int  # Short-axis searches run.
    initial_loss: float
    best_loss: float  # Of the saved correction.


def optimize_affines(
    subject_data: SubjectData,
    rounds: int = ROUNDS,
    search_radius: int = SEARCH_RADIUS,
    lax_steps: int = LAX_STEPS,
    lax_patience: int = LAX_PATIENCE,
    lr_lax: float = LR_LAX,
    seg_weight: float = SEG_WEIGHT,
    metric: str = METRIC,
    line_mm: float | None = LINE_MM,
    save: bool = True,
) -> AlignmentResult:
    r"""Alternates an exhaustive search of every short-axis slice's whole-voxel
    shift (within `search_radius` voxels of where it was acquired), long-axis
    planes fixed, with gradient descent on the long-axis rigid transforms,
    short-axis shifts fixed. Stops after `rounds` searches, or once a search
    no longer changes any shift.

    Where pairs are segmented on both sides, each search and refinement first
    uses the heart-overlap term, which has a wide basin, then intensities
    alone, within `FINE_RADIUS` voxels for the search: masks from separate
    segmentations disagree at their boundaries by about a voxel, which would
    otherwise bias the result by as much."""
    images = subject_data.images_pad
    segs = subject_data.segs_pad
    has_seg = subject_data.has_seg
    affines = subject_data.affines
    pair_indices = subject_data.idx_pairs.to(affines.device)
    shapes = subject_data.shapes
    spacings = subject_data.spacings
    device, dtype = affines.device, affines.dtype

    is_sa = torch.tensor(["sa" in n for n in subject_data.names], device=device)
    sa_planes = torch.where(is_sa)[0]
    la_planes = torch.where(~is_sa)[0]
    # Voxel sizes of the long-axis images, so their rotation is rigid in mm.
    la_scales = torch.linalg.norm(affines[la_planes, :3, :3], dim=1)
    # Pairs are (long axis, short axis) or (long axis, long axis).
    sa_la_pairs = pair_indices[is_sa[pair_indices[:, 1]]]
    slice_index = torch.full((len(affines),), -1, dtype=torch.long, device=device)
    slice_index[sa_planes] = torch.arange(len(sa_planes), device=device)
    pair_slice = slice_index[sa_la_pairs[:, 1]]
    coarse_offsets = shift_grid(search_radius, dtype, device)
    fine_offsets = shift_grid(FINE_RADIUS, dtype, device)
    uses_segs = seg_weight > 0 and bool((has_seg[pair_indices[:, 0]] & has_seg[pair_indices[:, 1]]).any())
    options = LossOptions(seg_weight, metric, line_mm)
    intensity_only = replace(options, seg_weight=0.0)
    stage_options = (options, intensity_only) if uses_segs else (intensity_only,)
    tensors = (images, segs, has_seg)
    geometry = (shapes, spacings)

    def transforms_for(sax_shifts, lax_params):
        return voxel_transformations(len(affines), sa_planes, la_planes, sax_shifts, lax_params, la_scales)

    def mean_loss(sax_shifts, lax_params):
        with torch.no_grad():
            loss = compute_pairwise_loss(
                *tensors, affines @ transforms_for(sax_shifts, lax_params), pair_indices, *geometry, options
            )
        return loss.mean().item()

    sax_shifts = torch.zeros((len(sa_planes), 2), dtype=dtype, device=device)
    lax_params = torch.zeros((len(la_planes), 6), dtype=dtype, device=device)
    initial_loss = mean_loss(sax_shifts, lax_params)
    stages, losses = ["initial"], [initial_loss]
    n_rounds = 0
    for n_rounds in range(1, rounds + 1):
        la_corrected = affines @ transforms_for(torch.zeros_like(sax_shifts), lax_params)
        with torch.no_grad():
            new_shifts = search_sax_shifts(
                *tensors, la_corrected, sa_planes, sa_la_pairs, pair_slice, *geometry, coarse_offsets, stage_options[0]
            )
            if uses_segs:
                new_shifts = search_sax_shifts(
                    *tensors, la_corrected, sa_planes, sa_la_pairs, pair_slice, *geometry, fine_offsets, intensity_only,
                    centres=new_shifts,
                )
        converged = n_rounds > 1 and torch.equal(new_shifts, sax_shifts)
        sax_shifts = new_shifts
        stages.append(f"sax search {n_rounds}")
        losses.append(mean_loss(sax_shifts, lax_params))
        if converged:
            break
        for stage in stage_options:
            lax_params = refine_lax(
                *tensors, affines, pair_indices, *geometry, sa_planes, la_planes,
                sax_shifts, lax_params, la_scales, lax_steps, lax_patience, lr_lax, stage,
            )
        stages.append(f"lax refinement {n_rounds}")
        losses.append(mean_loss(sax_shifts, lax_params))

    best_voxel_transformation = transforms_for(sax_shifts, lax_params).detach()
    best_loss = mean_loss(sax_shifts, lax_params)
    best_transformations = [None] * len(affines)
    for i, p in enumerate(sa_planes.tolist()):
        best_transformations[p] = sax_shifts[i]
    for p in la_planes.tolist():
        best_transformations[p] = best_voxel_transformation[p]
    tqdm.write(
        f"{subject_data.subject_path.name}: loss {initial_loss:.4g} -> {best_loss:.4g} in {n_rounds} rounds"
    )
    if save:
        subject_data.save_transformations(best_transformations)
    return AlignmentResult(
        transformations=best_transformations,
        voxel_transformations=best_voxel_transformation,
        stages=stages,
        losses=losses,
        rounds=n_rounds,
        initial_loss=initial_loss,
        best_loss=best_loss,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="Directory of subject directories")
    parser.add_argument("--rounds", type=int, default=ROUNDS, help="Maximum short-axis searches per subject")
    parser.add_argument(
        "--search_radius", type=int, default=SEARCH_RADIUS, help="Largest short-axis shift searched, in voxels"
    )
    parser.add_argument("--lax_steps", type=int, default=LAX_STEPS, help="Maximum Adam steps per long-axis refinement")
    parser.add_argument(
        "--lax_patience",
        type=int,
        default=LAX_PATIENCE,
        help="Stop a refinement after this many steps without improvement",
    )
    parser.add_argument("--lr_lax", type=float, default=LR_LAX, help="Adam learning rate of the long-axis transforms")
    parser.add_argument(
        "--seg_weight",
        type=float,
        default=SEG_WEIGHT,
        help="Weight of the heart-overlap term (0 to ignore segmentations)",
    )
    parser.add_argument("--metric", choices=METRICS, default=METRIC, help="Intensity dissimilarity along intersections")
    parser.add_argument(
        "--line_mm",
        type=float,
        default=LINE_MM,
        help="Length of each intersection compared, centred between the two images (0: half the largest image)",
    )
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
            rounds=args.rounds,
            search_radius=args.search_radius,
            lax_steps=args.lax_steps,
            lax_patience=args.lax_patience,
            lr_lax=args.lr_lax,
            seg_weight=args.seg_weight,
            metric=args.metric,
            line_mm=args.line_mm or None,
        )
    print(f"Corrected {len(subjects) - len(skipped)} of {len(subjects)} subjects in {args.data_dir}")
    if skipped:
        print(f"Skipped (no valid long-axis plane): {' '.join(skipped)}")
    print("Now rerun mrisr-register (slurm/register.sh): the corrections reach the model through it.")


if __name__ == "__main__":
    main()
