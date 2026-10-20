"""
UNet: the deterministic version of the Ladder VAE, trimmed from COSDD
(src/cosdd/networks/unet.py). Used as the direct predictor of E[x | y].

Changes from COSDD: separate input and output channels, plain
`torch.utils.checkpoint`, the final 1x1 convolution in float32, and the loss
left to the caller.

MIT License

Copyright (c) 2019 Andrea Dittadi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .nn import BottomUpLayer, Conv, ResBlockWithResampling, TopDownLayer
from .utils import get_padded_size, spatial_pad_crop


class UNet(nn.Module):
    """UNet model. Deterministic version of LVAE.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        blocks_per_layer (int): Number of residual blocks per layer.
        n_filters (int): Number of filters in the convolutional layers.
        n_layers (int): Number of layers in the UNet.
        td_skip (bool): Whether to use skip connections in the top-down pass.
        downsampling (list(list(int))): Per layer, halvings per dimension.
        scale_initialisation (bool): For stability, scale the last layer in the residual block by 1/(depth**0.5).
        checkpointed (bool): Whether to use activation checkpointing in the forward pass.
        dimensions (int): Dimensionality of the data (1, 2 or 3)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        blocks_per_layer=1,
        n_filters=64,
        n_layers=14,
        td_skip=True,
        downsampling=None,
        scale_initialisation=False,
        checkpointed=False,
        dimensions=2,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.blocks_per_layer = blocks_per_layer
        self.n_filters = n_filters
        self.td_skip = td_skip
        self.checkpointed = checkpointed

        assert downsampling is not None
        n_dim = len(downsampling[0])
        self.n_downsample = [0 for _ in range(n_dim)]
        for i in range(n_dim):
            for j in range(len(downsampling)):
                self.n_downsample[i] += downsampling[j][i]

        assert len(downsampling) == self.n_layers

        self.first_bottom_up = nn.Sequential(
            Conv(
                in_channels,
                n_filters,
                5,
                padding=2,
                padding_mode="replicate",
                dimensions=dimensions,
            ),
            nn.Mish(),
            ResBlockWithResampling(
                c_in=n_filters,
                c_out=n_filters,
                gated=False,
                n_resample=[0] * dimensions,
                scale_initialisation=scale_initialisation,
                dimensions=dimensions,
            ),
        )

        self.top_down_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()

        for i in range(self.n_layers):
            is_top = i == self.n_layers - 1

            self.bottom_up_layers.append(
                BottomUpLayer(
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=n_filters,
                    downsampling_steps=downsampling[i],
                    scale_initialisation=scale_initialisation,
                    dimensions=dimensions,
                )
            )

            self.top_down_layers.append(
                TopDownLayer(
                    n_res_blocks=blocks_per_layer,
                    n_filters=n_filters,
                    is_top_layer=is_top,
                    upsampling_steps=downsampling[i],
                    skip=td_skip,
                    scale_initialisation=scale_initialisation,
                    dimensions=dimensions,
                )
            )

        self.final_top_down = Conv(
            in_channels=n_filters,
            out_channels=out_channels,
            kernel_size=1,
            dimensions=dimensions,
        )

    def forward(self, x):
        # Pad x to have base 2 side lengths to make resampling steps simpler
        # Save size to crop back down later
        x_size = x.size()[2:]
        padded_size = get_padded_size(x_size, self.n_downsample)
        x = spatial_pad_crop(x, padded_size)

        bu_values = self.bottomup_pass(x)

        output = self.topdown_pass(bu_values)

        # Restore original image size
        output = spatial_pad_crop(output, x_size)
        return output

    def bottomup_pass(self, x):
        # Bottom-up initial layer
        x = self.first_bottom_up(x)

        # Loop from bottom to top layer, store all deterministic nodes we
        # need in the top-down pass
        bu_values = []
        for i in range(self.n_layers):
            if i % 2 == 0 and self.checkpointed and torch.is_grad_enabled():
                x = checkpoint(self.bottom_up_layers[i], x, use_reentrant=False)
            else:
                x = self.bottom_up_layers[i](x)
            bu_values.append(x)

        return bu_values

    def topdown_pass(self, bu_values):
        out = None
        for i in reversed(range(self.n_layers)):
            skip_input = out

            if i % 2 == 0 and self.checkpointed and torch.is_grad_enabled():
                out = checkpoint(
                    self.top_down_layers[i],
                    out,
                    skip_input,
                    bu_values[i],
                    use_reentrant=False,
                )
            else:
                out = self.top_down_layers[i](out, skip_input, bu_values[i])

        with torch.autocast(device_type=out.device.type, enabled=False):
            out = self.final_top_down(out.float())

        return out
