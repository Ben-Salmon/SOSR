r"""Network helpers, trimmed from COSDD (src/cosdd/networks/utils.py)."""

import math

import torch
from torch import nn


def norm_layer(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    r"""GroupNorm in place of COSDD's BatchRenorm.

    3D batches here hold one to four volumes, too few for batch statistics.
    """
    return nn.GroupNorm(math.gcd(channels, max_groups), channels)


def interleave(tensor1, tensor2):
    """
    Interleaves two tensors along the channel dimension.

    Args:
        tensor1 (torch.Tensor): The first input tensor.
        tensor2 (torch.Tensor): The second input tensor.

    Returns:
        torch.Tensor: The interleaved tensor.

    Raises:
        AssertionError: If the shapes of tensor1 and tensor2 do not match.
    """
    assert tensor1.shape == tensor2.shape, f"{tensor1.shape}, {tensor2.shape}"

    result = torch.stack((tensor1, tensor2), dim=2)

    N, C = result.shape[:2]
    result = result.reshape(N, 2 * C, *result.shape[3:])

    return result


def get_padded_size(size, n_downsc):
    """
    Calculates the necessary padded size of an image for a number of downscaling steps.

    Args:
        size (tuple): The desired size of the image.
        n_downsc (list(int)): The number of downscaling steps along each dimension.

    Returns:
        list: The padded size of the image.
    """
    dwnsc = [2**d for d in n_downsc]
    padded_size = [((s - 1) // d + 1) * d for s, d in zip(size, dwnsc)]

    return padded_size


def spatial_pad_crop(x, target_size):
    """
    Pads (with zeros) or crops the input tensor `x` to match the target size.

    Args:
        x (torch.Tensor): The input tensor to be padded or cropped.
        target_size (tuple): The target size to match.

    Returns:
        torch.Tensor: The padded or cropped tensor.
    """
    x_size = x.size()[2:]
    delta = [ts - xs for ts, xs in zip(target_size, x_size)]
    crop_delta = [(abs(d) // 2, abs(d) // 2 + abs(d) % 2) if d < 0 else (0, 0) for d in delta]
    pad_delta = [(d // 2, d // 2 + d % 2) if d > 0 else (0, 0) for d in delta]

    pad = []
    for d in reversed(pad_delta):
        pad.append(d[0])
        pad.append(d[1])
    x = nn.functional.pad(x, pad)
    x_size = x.size()[2:]
    crop = [slice(0, x.size(0)), slice(0, x.size(1))]
    crop += [slice(d[0], xs - d[1]) for d, xs in zip(crop_delta, x_size)]
    return x[tuple(crop)]


class LinearUpsample(nn.Module):
    """
    Upsamples the input tensor `x` using linear/bilinear/trilinear interpolation.

    Args:
        scale_factor (int or tuple): The scale factor for the upsampling operation.
    """

    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.modes = {1: "linear", 2: "bilinear", 3: "trilinear"}

    def forward(self, x):
        d = x.dim() - 2
        if all(s == 1 for s in self.scale_factor):
            return x
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.modes[d])
