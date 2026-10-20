r"""The MRI forward model: latent volume -> observed views.

The Lanczos resampling kernels are copied unchanged from
diffusion-priors/forward_model.py. What differs is the glue around them:

- `build_view_operator` is the body of `ForwardModel.set_operations`, with the
  latent geometry passed in instead of read from module globals. It returns
  plain CPU tensors, so DataLoader workers can build operators in parallel.
- `lanczos_adjoint` is the body of `FusedLanczosOp.backward`, so that the
  backward pass and the explicit adjoint (`FusedLanczosResample.adjoint`) are
  the same code and stay exact transposes.
- `BatchOperator` replaces `ForwardModel`. It returns one tensor per
  observation and view, with no padding or flattening, since the VAE sums its
  likelihood view by view.
"""

import math

import torch
from torch import Tensor, meshgrid, nn
from torch.amp import custom_bwd, custom_fwd


def invert_affine(affine):
    inv_affine = torch.linalg.inv(affine)
    inv_affine[3, :3] = 0  # correct inv errors
    return inv_affine


def apply_affine(affine, coords):
    if coords.ndim == 1:
        coords = coords[:, None]
        remove_extra = True
    else:
        remove_extra = False
    coords = torch.cat((coords, torch.ones_like(coords[:1])), 0)
    coords = affine @ coords
    coords = coords[:3]
    if remove_extra:
        coords = coords[:, 0]
    return coords


def get_vox_su_affine(observation, view):
    vox_to_mm_affine = torch.as_tensor(observation[view]["vox_to_mm_affine"]).to(torch.float32)
    mm_to_su_affine = torch.as_tensor(observation["mm_to_su_affine"]).to(torch.float32)
    vox_to_su_affine = mm_to_su_affine @ vox_to_mm_affine
    su_to_vox_affine = invert_affine(vox_to_su_affine)
    return vox_to_su_affine, su_to_vox_affine


def bound_y_coords(x_coords_vox, y_size_vox):
    """
    Returns [y_min, y_max) (slice convention): the y-voxels whose centres lie
    within the bounding box of the latent voxel centres, in y-voxel space.

    This is a crop, not a support test. For an oblique view the box holds
    y-voxels outside the rotated latent, and near the box faces a kernel
    footprint (Lanczos taps, slice profile) can leave the latent.
    `build_view_operator` keeps those rows out of the likelihood.
    """
    x_min_coords_vox = x_coords_vox.min(dim=1)[0]
    x_max_coords_vox = x_coords_vox.max(dim=1)[0]
    zeros = torch.zeros_like(x_min_coords_vox)
    y_min_coords_vox = torch.maximum(torch.ceil(x_min_coords_vox + 0.5), zeros)
    y_max_coords_vox = torch.minimum(
        torch.floor(x_max_coords_vox - 0.5) + 1, y_size_vox
    )  # +1 fixes off-by-one
    y_max_coords_vox = torch.maximum(
        y_max_coords_vox, y_min_coords_vox
    )  # guard against degenerate (zero-size) outputs
    return y_max_coords_vox, y_min_coords_vox


def get_x_kernel_size(mm_per_vox, su_per_lvox):
    x_kernel_size_lvox = torch.tensor([1, 1, 1])
    x_kernel_size_mm = su_per_lvox * x_kernel_size_lvox
    x_kernel_size_vox = x_kernel_size_mm / mm_per_vox
    return x_kernel_size_vox


def apply_gaussian_slice_profile(kernel, fwhm):
    beta = 2
    sigma = fwhm / (2 * math.log(2) ** (1 / beta))
    slice_profile = (
        torch.arange(kernel.shape[2], dtype=torch.float32) - (kernel.shape[2] - 1) / 2
    ).reshape(1, 1, -1)
    slice_profile = torch.exp(-(torch.abs(slice_profile) ** beta) / (sigma**beta))
    kernel *= slice_profile
    return kernel


def lanczos_kernel_1d(t, a):
    return torch.where(
        torch.abs(t) < a,
        torch.sinc(t) * torch.sinc(t / a),
        torch.zeros_like(t),
    )


def _lanczos_weights(positions, a, sizes_long):
    """
    Lanczos-a weights and lattice indices for sample positions, shared by the
    forward and backward passes so that they stay exact adjoints.

    Along each axis the 2a weights are normalised to sum to one before
    out-of-range taps are zeroed. Unnormalised, Lanczos-3 reproduces a
    constant only to within ~0.6% per axis, varying with the sub-voxel
    position. A sample whose taps leave the volume no longer sums to one;
    those are kept out of the likelihood (see `lanczos_support_mask`).

    positions:  (N, M, 3) lvox coordinates ordered (d, h, w). Kept in float32
                whatever the volume's dtype: in bfloat16 a coordinate near 80
                would be rounded to the nearest 0.5.
    sizes_long: long buffer of shape (1, 1, 3, 1) holding [D, H, W].
    Returns:    weights (N, M, 3, 2a) in float32 and indices (N, M, 3, 2a).
    """
    K = 2 * a
    positions = positions.to(torch.float32)
    floor_pos = torch.floor(positions)  # (N, M, 3)
    frac = positions - floor_pos  # (N, M, 3)
    offsets = (
        torch.arange(K, device=positions.device, dtype=torch.float32) - (a - 1)
    )  # (K,)

    dist = offsets.view(1, 1, 1, K) - frac.unsqueeze(-1)  # (N, M, 3, K)
    w = lanczos_kernel_1d(dist, a)
    w = w / w.sum(dim=-1, keepdim=True)

    idx = floor_pos.long().unsqueeze(-1) + offsets.long().view(
        1, 1, 1, K
    )  # (N, M, 3, K)
    valid = (idx >= 0) & (idx < sizes_long)
    w = w * valid
    idx = torch.clamp(idx, min=0)
    idx = torch.minimum(idx, sizes_long - 1)
    return w, idx


def lanczos_support_mask(positions, a, sizes):
    """
    Whether every tap of a Lanczos-a sample lies inside the volume, i.e.
    whether `_lanczos_weights` keeps all of them.

    positions: (..., 3) lvox coordinates.
    sizes:     (3,) float tensor holding [D, H, W].
    Returns:   (...) bool.
    """
    floor_pos = torch.floor(positions.to(torch.float32))
    return ((floor_pos >= a - 1) & (floor_pos <= sizes - 1 - a)).all(dim=-1)


def _lanczos_sample_3d_impl(x, positions, a, sizes_long, hw, w_stride):
    """
    sizes_long: long buffer of shape (1, 1, 3, 1) holding [D, H, W].
    hw:         int H * W.
    w_stride:   int W.
    """
    N, C = x.shape[:2]
    M = positions.shape[1]
    K = 2 * a

    w, idx = _lanczos_weights(positions, a, sizes_long)
    w = w.to(x.dtype)

    w_d, w_h, w_w = w[..., 0, :], w[..., 1, :], w[..., 2, :]  # each (N, M, K)
    idx_d, idx_h, idx_w = idx[..., 0, :], idx[..., 1, :], idx[..., 2, :]

    flat_idx = (
        idx_d[..., :, None, None] * hw
        + idx_h[..., None, :, None] * w_stride
        + idx_w[..., None, None, :]
    )  # (N, M, K, K, K)
    flat_idx = flat_idx.reshape(N, M * K * K * K)

    x_flat = x.reshape(N, C, -1)
    gathered = torch.gather(x_flat, 2, flat_idx.unsqueeze(1).expand(N, C, -1)).reshape(
        N, C, M, K, K, K
    )

    return torch.einsum("nmi,nmj,nmk,ncmijk->ncm", w_d, w_h, w_w, gathered)


def lanczos_sample_3d(x, positions, sizes_long, hw, w_stride, a=3, chunk_size=16384):
    """
    Lanczos-a interpolation of a 3D volume at arbitrary positions, with zero
    padding outside the volume. Chunks over query points to bound peak memory
    of the (M, K^3) gather.

    x:          (N, C, D, H, W)
    positions:  (N, M, 3) lvox coordinates ordered (d, h, w).
    sizes_long: long buffer of shape (1, 1, 3, 1) holding [D, H, W].
    hw:         int H * W.
    w_stride:   int W.
    Returns:    (N, C, M)
    """
    N, C = x.shape[:2]
    M = positions.shape[1]
    out = x.new_empty(N, C, M)
    for s in range(0, M, chunk_size):
        e = min(s + chunk_size, M)
        out[..., s:e] = _lanczos_sample_3d_impl(
            x, positions[:, s:e], a, sizes_long, hw, w_stride
        )
    return out


def _scatter_lanczos_grad(
    grad_sample, positions, dtype, a, sizes_long, hw, w_stride, grad_x_flat, chunk_size
):
    """
    Inverse of one kernel-position's gather/einsum. For each query (m, lattice
    neighbour), scatter-add weight * grad_sample[c, m] into grad_x_flat at the
    flat lattice index. Mirrors `_lanczos_sample_3d_impl` on the index/weight
    side; no gathered values are materialised.

    grad_sample: (N, C, M) — gradient w.r.t. lanczos sample values.
    positions:   (1, M, 3) — same lvox positions the forward used.
    grad_x_flat: (N, C, D*H*W) — accumulated into in place.
    """
    N, C, M = grad_sample.shape

    for s in range(0, M, chunk_size):
        e = min(s + chunk_size, M)
        pos = positions[:, s:e]  # (1, chunk, 3)

        # Exactly the forward pass's weights, normalisation included.
        w, idx = _lanczos_weights(pos, a, sizes_long)  # (1, chunk, 3, K)

        w_d, w_h, w_w = w[..., 0, :], w[..., 1, :], w[..., 2, :]
        idx_d, idx_h, idx_w = idx[..., 0, :], idx[..., 1, :], idx[..., 2, :]

        # Separable -> (1, chunk, K, K, K) outer-product weight tensor.
        weight = (
            w_d[..., :, None, None] * w_h[..., None, :, None] * w_w[..., None, None, :]
        )
        flat_idx = (
            idx_d[..., :, None, None] * hw
            + idx_h[..., None, :, None] * w_stride
            + idx_w[..., None, None, :]
        )  # (1, chunk, K, K, K)

        # Project grad_sample onto neighbours: (N, C, chunk, K, K, K)
        chunk_grad = grad_sample[..., s:e]  # (N, C, chunk)
        contrib = weight[:, None] * chunk_grad[..., None, None, None]
        flat_idx_b = flat_idx[:, None].expand_as(contrib).reshape(N, C, -1)
        grad_x_flat.scatter_add_(
            2, flat_idx_b, contrib.reshape(N, C, -1).to(grad_x_flat.dtype)
        )


def lanczos_adjoint(
    grad_y_hat,
    y_centres_lvox,
    kernel_offsets_lvox,
    combined_kernel,
    sizes_long,
    hw,
    w_stride,
    a,
    chunk_size,
    n_shape,
    out_dtype,
):
    """
    Transpose of the Lanczos resample + per-y-voxel kernel sum: the body of
    `FusedLanczosOp.backward`, also used for the explicit adjoint.

    grad_y_hat: (N, C, n_y0, n_y1, n_y2)
    Returns:    (N, C, D, H, W) in `out_dtype`.
    """
    n_y0, n_y1, n_y2 = n_shape
    N, C = grad_y_hat.shape[:2]
    D, H, W = sizes_long.view(3).tolist()
    M = n_y0 * n_y1 * n_y2
    y_centres_flat = y_centres_lvox.reshape(M, 3)

    grad_x = torch.zeros(N, C, D, H, W, dtype=out_dtype, device=grad_y_hat.device)
    grad_x_flat = grad_x.view(N, C, D * H * W)

    for k in range(combined_kernel.shape[0]):
        kernel_w = combined_kernel[k].view(1, 1, 1, 1, n_y2)
        grad_sample = (grad_y_hat * kernel_w).reshape(N, C, M)
        positions = (y_centres_flat + kernel_offsets_lvox[k]).unsqueeze(0)
        _scatter_lanczos_grad(
            grad_sample,
            positions,
            out_dtype,
            a,
            sizes_long,
            hw,
            w_stride,
            grad_x_flat,
            chunk_size,
        )
    return grad_x


class FusedLanczosOp(torch.autograd.Function):
    """
    Custom autograd op for the Lanczos-3D resample + per-y-voxel kernel sum.

    Forward saves only `x` (plus references to the constant buffers) — none of
    the per-chunk gather, einsum, or index intermediates persist for backward.
    Backward recomputes the (cheap) Lanczos weights and lattice indices and
    scatters the projected cotangent directly into grad_x. Compared to relying
    on autograd's automatic backward through gather/einsum, this drops peak
    memory from K_total * N_chunks of saved activations to one chunk's worth,
    without the recompute cost of gradient checkpointing.
    """

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x,
        y_centres_lvox,
        kernel_offsets_lvox,
        combined_kernel,
        sizes_long,
        hw,
        w_stride,
        a,
        chunk_size,
        n_shape,
    ):
        ctx.save_for_backward(x)
        ctx.y_centres_lvox = y_centres_lvox
        ctx.kernel_offsets_lvox = kernel_offsets_lvox
        ctx.combined_kernel = combined_kernel
        ctx.sizes_long = sizes_long
        ctx.hw = hw
        ctx.w_stride = w_stride
        ctx.a = a
        ctx.chunk_size = chunk_size
        ctx.n_shape = n_shape

        n_y0, n_y1, n_y2 = n_shape
        N, C = x.shape[:2]
        M = n_y0 * n_y1 * n_y2
        y_centres_flat = y_centres_lvox.reshape(M, 3)

        y_hat = x.new_zeros(N, C, n_y0, n_y1, n_y2)
        for k in range(combined_kernel.shape[0]):
            positions = (y_centres_flat + kernel_offsets_lvox[k]).unsqueeze(0)
            sample = lanczos_sample_3d(
                x,
                positions,
                sizes_long=sizes_long,
                hw=hw,
                w_stride=w_stride,
                a=a,
                chunk_size=chunk_size,
            )
            sample = sample.reshape(N, C, n_y0, n_y1, n_y2)
            y_hat = y_hat + sample * combined_kernel[k].view(1, 1, 1, 1, n_y2)
        return y_hat

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_y_hat):
        (x,) = ctx.saved_tensors
        grad_x = lanczos_adjoint(
            grad_y_hat,
            ctx.y_centres_lvox,
            ctx.kernel_offsets_lvox,
            ctx.combined_kernel,
            ctx.sizes_long,
            ctx.hw,
            ctx.w_stride,
            ctx.a,
            ctx.chunk_size,
            ctx.n_shape,
            out_dtype=x.dtype,
        )
        # Gradients for: x, y_centres, kernel_offsets, combined_kernel,
        # sizes_long, hw, w_stride, a, chunk_size, n_shape.
        return (grad_x,) + (None,) * 9


class FusedLanczosResample(nn.Module):
    """
    Lanczos-3D resample of x at sub-voxel positions with weighted accumulation
    into y-voxels. Replaces grid_sample(trilinear) -> zero-pad -> conv3d, never
    materialising the full sub-sampled grid.

    Backed by FusedLanczosOp, a custom autograd.Function whose backward
    computes grad_x by scatter-add of recomputed Lanczos weights — so neither
    the gather nor the einsum activations are saved for backward.
    """

    def __init__(
        self,
        y_centres_lvox,
        kernel_offsets_lvox,
        combined_kernel,
        row_mask,
        x_shape,
        lanczos_a=3,
        chunk_size=16384,
    ):
        super().__init__()
        self.lanczos_a = lanczos_a
        self.chunk_size = chunk_size
        # x's spatial shape is fixed by the global latent geometry, so we can
        # bake D, H, W and the row strides into the module instead of reading
        # them from x.shape on every call.
        D, H, W = (int(s) for s in x_shape)
        self.D = D
        self.H = H
        self.W = W
        self.hw = H * W
        self.register_buffer("y_centres_lvox", y_centres_lvox)  # (n_y0, n_y1, n_y2, 3)
        self.register_buffer("kernel_offsets_lvox", kernel_offsets_lvox)  # (K_total, 3)
        # Kernel weight per sample position and y-slice.
        self.register_buffer("combined_kernel", combined_kernel)  # (K_total, n_y2)
        # One on the y-voxels whose whole footprint lies inside the latent; the
        # rest are excluded from the likelihood.
        self.register_buffer("row_mask", row_mask.to(torch.float32))  # (n_y0, n_y1, n_y2)
        self.register_buffer(
            "sizes_long",
            torch.tensor([D, H, W], dtype=torch.long).view(1, 1, 3, 1),
        )

    @property
    def n_shape(self):
        return tuple(int(s) for s in self.y_centres_lvox.shape[:3])

    def forward(self, x: Tensor):
        assert x.shape[-3:] == (self.D, self.H, self.W), (
            f"x spatial shape {tuple(x.shape[-3:])} != expected "
            f"{(self.D, self.H, self.W)}"
        )
        return FusedLanczosOp.apply(
            x,
            self.y_centres_lvox,
            self.kernel_offsets_lvox,
            self.combined_kernel,
            self.sizes_long,
            self.hw,
            self.W,
            self.lanczos_a,
            self.chunk_size,
            self.n_shape,
        )

    def adjoint(self, v: Tensor) -> Tensor:
        """
        Adjoint of the masked operator `row_mask * self(x)`.

        v:       (N, C, n_y0, n_y1, n_y2)
        Returns: (N, C, D, H, W)
        """
        assert tuple(v.shape[-3:]) == self.n_shape, (
            f"v spatial shape {tuple(v.shape[-3:])} != operator output {self.n_shape}"
        )
        v = v * self.row_mask.to(v.dtype)
        return lanczos_adjoint(
            v,
            self.y_centres_lvox,
            self.kernel_offsets_lvox,
            self.combined_kernel,
            self.sizes_long,
            self.hw,
            self.W,
            self.lanczos_a,
            self.chunk_size,
            self.n_shape,
            out_dtype=v.dtype,
        )


def build_view_operator(observation: dict, view: str, geometry, lanczos_a: int = 3):
    """
    The forward operator of one view, as plain CPU tensors.

    The body of `ForwardModel.set_operations` in diffusion-priors, with the
    latent geometry passed in. `observation` holds `mm_to_su_affine` and, under
    `view`, `vox_to_mm_affine`, `mm_per_vox`, `kernel_size_mm` and `size_vox`.

    Returns the keyword arguments of `FusedLanczosResample` other than the
    latent shape -- `y_centres_lvox`, `kernel_offsets_lvox`, `combined_kernel`
    and `row_mask` -- or None if the view does not overlap the latent.
    """
    y_size_vox = torch.as_tensor(observation[view]["size_vox"])
    mm_per_vox = torch.abs(torch.as_tensor(observation[view]["mm_per_vox"]).to(torch.float32))
    y_kernel_size_mm = torch.as_tensor(observation[view]["kernel_size_mm"])

    y_kernel_size_vox = y_kernel_size_mm / mm_per_vox
    vox_to_su_affine, su_to_vox_affine = get_vox_su_affine(observation, view)

    x_coords_su_flat = torch.flatten(geometry.coords_su, start_dim=1, end_dim=-1)
    x_coords_vox = apply_affine(affine=su_to_vox_affine, coords=x_coords_su_flat)
    y_max_coords_vox, y_min_coords_vox = bound_y_coords(x_coords_vox, y_size_vox)
    size_voxels = y_max_coords_vox - y_min_coords_vox
    n_y = [int(s) for s in size_voxels]
    if 0 in n_y:
        return None

    x_kernel_size_vox = get_x_kernel_size(mm_per_vox, geometry.su_per_lvox)
    x_samples_per_y_samples = 1 / x_kernel_size_vox
    rounded_x_samples_per_y_samples = torch.ceil(x_samples_per_y_samples)

    su_to_lvox_affine = invert_affine(geometry.lvox_to_su_affine)
    vox_to_lvox_affine = su_to_lvox_affine @ vox_to_su_affine

    n_sub = [int(r) for r in rounded_x_samples_per_y_samples]
    slice_overhang = 2

    # y-voxel centres in vox -> lvox.
    y_axes_vox = [
        y_min_coords_vox[d] + torch.arange(n_y[d], dtype=torch.float32)
        for d in range(3)
    ]
    y_centres_vox = torch.stack(meshgrid(*y_axes_vox, indexing="ij"), dim=0)
    y_centres_vox_flat = torch.flatten(y_centres_vox, start_dim=1)
    y_centres_lvox_flat = apply_affine(vox_to_lvox_affine, y_centres_vox_flat)
    y_centres_lvox = torch.unflatten(
        y_centres_lvox_flat, dim=1, sizes=y_centres_vox.shape[1:]
    )
    y_centres_lvox = torch.movedim(y_centres_lvox, 0, -1).contiguous()

    # Kernel sub-offsets (vox), then transformed to lvox by linear part of
    # vox_to_lvox. Slice axis (axis 2) carries the conv overhang.
    ds_vox = 1 / rounded_x_samples_per_y_samples
    sub_offsets_axes_vox = []
    for d in range(3):
        extra = 2 * slice_overhang if d == 2 else 0
        shift = slice_overhang if d == 2 else 0
        n_full = n_sub[d] + extra
        indices = torch.arange(n_full, dtype=torch.float32)
        offs = (indices - shift + 0.5) * ds_vox[d] - 0.5
        sub_offsets_axes_vox.append(offs)
    sub_offsets_vox = torch.stack(
        meshgrid(*sub_offsets_axes_vox, indexing="ij"), dim=-1
    )
    K_shape = sub_offsets_vox.shape[:3]
    K_total = K_shape[0] * K_shape[1] * K_shape[2]
    sub_offsets_vox_flat = sub_offsets_vox.reshape(K_total, 3).T
    vox_to_lvox_linear = vox_to_lvox_affine[:3, :3]
    kernel_offsets_lvox = (vox_to_lvox_linear @ sub_offsets_vox_flat).T.contiguous()

    # Kernel weights: gaussian along slice axis * ones in-plane, normalised.
    y_kernel = torch.ones(K_shape)
    fwhm = y_kernel_size_vox[2] * rounded_x_samples_per_y_samples[2]
    y_kernel = apply_gaussian_slice_profile(y_kernel, fwhm)
    y_kernel = y_kernel / y_kernel.sum()
    kernel_weights = y_kernel.reshape(K_total)

    # The same kernel for every slice. The overhang samples are read from x
    # like any others; where they leave the latent, the row is excluded below.
    combined_kernel = kernel_weights[:, None].expand(K_total, n_y[2]).contiguous()

    # A y-voxel is modelled exactly only if every Lanczos tap of every kernel
    # sample lies inside the latent. Near the latent's faces the footprint is
    # cut short and the prediction biased, and for an oblique view the crop
    # also holds voxels entirely outside the rotated latent, which nothing in
    # x can explain. Those rows are left out of the likelihood rather than
    # fitted.
    sizes = torch.tensor(geometry.size_lvox, dtype=torch.float32)
    row_mask = torch.ones(y_centres_lvox.shape[:3], dtype=torch.bool)
    for offset in kernel_offsets_lvox:
        row_mask &= lanczos_support_mask(y_centres_lvox + offset, lanczos_a, sizes)

    return {
        "y_centres_lvox": y_centres_lvox,
        "kernel_offsets_lvox": kernel_offsets_lvox,
        "combined_kernel": combined_kernel,
        "row_mask": row_mask,
    }


class BatchOperator(nn.Module):
    """
    The forward operators of a batch: one per observation and view.

    `operators[b]` maps each view of observation b to the output of
    `build_view_operator`, in the order the views should be returned.
    Build it on the CPU and move it with `.to(device)`.
    """

    def __init__(self, operators, size_lvox, lanczos_a=3, chunk_size=16384):
        super().__init__()
        self.size_lvox = tuple(int(s) for s in size_lvox)
        self.views = [list(ops.keys()) for ops in operators]
        self.ops = nn.ModuleList(
            nn.ModuleDict(
                {
                    view: FusedLanczosResample(
                        x_shape=self.size_lvox,
                        lanczos_a=lanczos_a,
                        chunk_size=chunk_size,
                        **op,
                    )
                    for view, op in ops.items()
                }
            )
            for ops in operators
        )

    def __len__(self):
        return len(self.ops)

    def op(self, b: int, view: str) -> FusedLanczosResample:
        return self.ops[b][view]

    def forward(self, x: Tensor) -> list[list[Tensor]]:
        """
        x:       (B, C, D, H, W)
        Returns: per observation, per view, (1, C, n_y0, n_y1, n_y2), zero on
                 the rows the latent does not support (`row_mask`).
        """
        assert x.shape[0] == len(self.ops), f"{x.shape[0]} latents for {len(self.ops)} observations"
        x = x.contiguous()
        out = []
        for b, views in enumerate(self.views):
            predictions = []
            for view in views:
                op = self.ops[b][view]
                y_hat = op(x[b : b + 1]).to(x.dtype)
                # The adjoint, taken by autograd, picks the mask up too.
                predictions.append(y_hat * op.row_mask.to(y_hat.dtype))
            out.append(predictions)
        return out

    def adjoint(self, v: list[list[Tensor]]) -> Tensor:
        """
        Adjoint of `forward`.

        v:       per observation, per view, (1, C, n_y0, n_y1, n_y2).
        Returns: (B, C, D, H, W)
        """
        first = v[0][0]
        out = torch.zeros(
            len(self.ops), first.shape[1], *self.size_lvox, dtype=first.dtype, device=first.device
        )
        for b, views in enumerate(self.views):
            for view, vb in zip(views, v[b], strict=True):
                out[b : b + 1] += self.ops[b][view].adjoint(vb)
        return out
