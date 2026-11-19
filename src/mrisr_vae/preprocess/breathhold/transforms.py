import torch


def params_to_mat(params: torch.Tensor) -> torch.Tensor:
    r"""(n, 6) Euler angles (z, y, x; radians) and translations -> (n, 4, 4)."""
    assert params.shape[1] == 6
    rotation, translation = params[:, :3], params[:, 3:]
    cos = torch.cos(rotation)
    sin = torch.sin(rotation)
    rotation_z = torch.eye(3, dtype=params.dtype, device=params.device).repeat((params.shape[0], 1, 1))
    rotation_z[:, 0, 0] = cos[:, 0]
    rotation_z[:, 0, 1] = -sin[:, 0]
    rotation_z[:, 1, 0] = sin[:, 0]
    rotation_z[:, 1, 1] = cos[:, 0]

    rotation_y = torch.eye(3, dtype=params.dtype, device=params.device).repeat((params.shape[0], 1, 1))
    rotation_y[:, 0, 0] = cos[:, 1]
    rotation_y[:, 0, 2] = sin[:, 1]
    rotation_y[:, 2, 0] = -sin[:, 1]
    rotation_y[:, 2, 2] = cos[:, 1]

    rotation_x = torch.eye(3, dtype=params.dtype, device=params.device).repeat((params.shape[0], 1, 1))
    rotation_x[:, 1, 1] = cos[:, 2]
    rotation_x[:, 1, 2] = -sin[:, 2]
    rotation_x[:, 2, 1] = sin[:, 2]
    rotation_x[:, 2, 2] = cos[:, 2]

    rotation = torch.bmm(rotation_x, rotation_y)
    rotation = torch.bmm(rotation, rotation_z)

    affines = torch.eye(4, dtype=params.dtype, device=params.device).repeat((params.shape[0], 1, 1))
    affines[:, :3, :3] = rotation
    affines[:, :3, 3] = translation
    return affines


def lax_voxel_transform(params: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    r"""A rigid long-axis correction, as a voxel-space transform.

    The rotation of `params_to_mat` is conjugated by the voxel sizes,
    S^{-1} R S, so that `affine @ transform` stays rigid in mm. Applied directly
    in voxel units, any tilt out of the image plane mixes the in-plane pixel
    size with the much larger slice thickness and becomes a shear, which
    nibabel then strips when the corrected affine is written to the qform that
    `mrisr-register` reads. Translations stay in voxel units.

    params: (n, 6) rotation (radians) and translation (voxels).
    scales: (n, 3) voxel sizes, the column norms of each image's affine.
    """
    transform = params_to_mat(params)
    rotation = transform[:, :3, :3]
    rotation = rotation / scales[:, :, None] * scales[:, None, :]
    top = torch.cat((rotation, transform[:, :3, 3:]), dim=2)
    return torch.cat((top, transform[:, 3:, :]), dim=1)
