import torch


def fast_trilinear_interpolation(
    input_array: torch.Tensor,
    y_indices: torch.Tensor,
    x_indices: torch.Tensor,
    z_indices: torch.Tensor,
) -> torch.Tensor:
    """Trilinear interpolation of a batch of 3D volumes.

    :param input_array: Images used as source for the sampling.               Shape: (batch, height, width, depth)
    :param y_indices: Indices of the 1st spatial dimension of a given image.  Shape: (batch, num_points)
    :param x_indices: Indices of the 2nd spatial dimension.                   Shape: (batch, num_points)
    :param z_indices: Indices of the 3rd spatial dimension.                   Shape: (batch, num_points)
    """
    x0 = torch.floor(y_indices.detach()).to(torch.long)
    y0 = torch.floor(x_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[1] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[2] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[3] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[1] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[2] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[3] - 1)

    x = y_indices - x0
    y = x_indices - y0
    z = z_indices - z0

    b, _ = torch.meshgrid(
        torch.arange(0, x.shape[0], device=x.device),
        torch.arange(0, x.shape[1], device=x.device),
        indexing="ij",
    )
    b_ = b.reshape(-1)
    x0_ = x0.reshape(-1)
    x1_ = x1.reshape(-1)
    y0_ = y0.reshape(-1)
    y1_ = y1.reshape(-1)
    z0_ = z0.reshape(-1)
    z1_ = z1.reshape(-1)
    x_ = x.reshape(-1)
    y_ = y.reshape(-1)
    z_ = z.reshape(-1)
    output_ = (
        input_array[b_, x0_, y0_, z0_] * (1 - x_) * (1 - y_) * (1 - z_)
        + input_array[b_, x1_, y0_, z0_] * x_ * (1 - y_) * (1 - z_)
        + input_array[b_, x0_, y1_, z0_] * (1 - x_) * y_ * (1 - z_)
        + input_array[b_, x0_, y0_, z1_] * (1 - x_) * (1 - y_) * z_
        + input_array[b_, x1_, y0_, z1_] * x_ * (1 - y_) * z_
        + input_array[b_, x0_, y1_, z1_] * (1 - x_) * y_ * z_
        + input_array[b_, x1_, y1_, z0_] * x_ * y_ * (1 - z_)
        + input_array[b_, x1_, y1_, z1_] * x_ * y_ * z_
    )
    return output_.reshape(x0.shape)
