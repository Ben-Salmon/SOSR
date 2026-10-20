"""
Ladder VAE building blocks, trimmed from COSDD (src/cosdd/networks/nn.py).

Changes from COSDD: GroupNorm instead of BatchRenorm, no 4D convolutions or
PixelCNN parts, and the stochastic layer's parameters are computed in float32.

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

from .utils import LinearUpsample, interleave, norm_layer


class Conv(nn.Module):
    """
    Convolutional layer for 1, 2, or 3D data.

    Same arguments as nn.Convxd but dimensionality of data is
    passed as argument.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        dimensions=2,
    ):
        super().__init__()
        if dimensions not in (1, 2, 3):
            raise NotImplementedError(f"{dimensions}D convolutions")
        conv = getattr(nn, f"Conv{dimensions}d")
        self.conv = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        return self.conv(x)


class GateLayer(nn.Module):
    def __init__(self, channels, kernel_size, dimensions=2):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        self.conv = Conv(channels, 2 * channels, kernel_size, padding=pad, dimensions=dimensions)
        self.nonlin = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x, gate = torch.chunk(x, dim=1, chunks=2)
        x = self.nonlin(x)
        gate = torch.sigmoid(gate)
        return x * gate


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        groups=1,
        gated=True,
        scale_initialisation=False,
        dimensions=2,
    ):
        super().__init__()
        self.channels = channels
        assert kernel_size % 2 == 1
        self.pad = kernel_size // 2
        self.kernel_size = kernel_size
        self.groups = groups
        self.gated = gated
        if scale_initialisation:
            self.rescale = channels**0.5
        else:
            self.rescale = 1.0

        self.block = nn.Sequential()
        for _ in range(2):
            self.block.append(norm_layer(self.channels))
            self.block.append(nn.ReLU())
            conv = Conv(
                self.channels,
                self.channels,
                self.kernel_size,
                padding=self.pad,
                padding_mode="reflect",
                groups=self.groups,
                dimensions=dimensions,
            )
            self.block.append(conv)
        if self.gated:
            self.block.append(GateLayer(self.channels, 1, dimensions=dimensions))

    def forward(self, x):
        block_out = self.block(x) / self.rescale
        out = block_out + x
        return out


class ResBlockWithResampling(nn.Module):
    """
    Residual Block with Resampling.

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        resample (str): Resampling method. Can be "up", "down", or None.
        n_resample (list(int)): Number of halvings/doublings per dimension.
        res_block_kernel (int): Kernel size for the residual block.
        groups (int): Number of groups for grouped convolution.
        gated (bool): Whether to use gated activation.
        scale_initialisation (bool): For stability, scale the last layer in the residual block by 1/(depth**0.5).
            See VDVAE, Child 2020.
        dimensions (int): Dimensionality of the data (1, 2 or 3).
    """

    def __init__(
        self,
        c_in,
        c_out,
        resample=None,
        n_resample=None,
        res_block_kernel=3,
        groups=1,
        gated=True,
        scale_initialisation=False,
        dimensions=2,
    ):
        super().__init__()
        assert resample in ["up", "down", None]
        scale_factor = [2**n for n in n_resample]

        if resample == "up":
            self.pre_conv = nn.Sequential(
                LinearUpsample(scale_factor=scale_factor),
                Conv(c_in, c_out, 1, groups=groups, padding_mode="reflect", dimensions=dimensions),
            )
        elif resample == "down":
            self.pre_conv = Conv(
                c_in,
                c_out,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                stride=scale_factor,
                groups=groups,
                dimensions=dimensions,
            )
        elif c_in != c_out:
            self.pre_conv = Conv(c_in, c_out, 1, groups=groups, padding_mode="reflect", dimensions=dimensions)
        else:
            self.pre_conv = nn.Identity()

        self.res_block = ResidualBlock(
            channels=c_out,
            kernel_size=res_block_kernel,
            groups=groups,
            gated=gated,
            scale_initialisation=scale_initialisation,
            dimensions=dimensions,
        )

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.res_block(x)
        return x


class MergeLayer(nn.Module):
    """
    Merges two input tensors and applies a convolutional layer followed by a residual block.

    Args:
        channels (int or list[int]): The number of channels of the two inputs and the output.
        scale_initialisation (bool): For stability, scale the last layer in the residual block by 1/(depth**0.5).
            See VDVAE, Child 2020.
        dimensions (int): Dimensionality of the data (1, 2 or 3)
    """

    def __init__(self, channels, scale_initialisation=False, dimensions=2):
        super().__init__()
        try:
            iter(channels)
        except TypeError:  # it is not iterable
            channels = [channels] * 3
        else:  # it is iterable
            if len(channels) == 1:
                channels = [channels[0]] * 3
        assert len(channels) == 3

        self.layer = nn.Sequential(
            Conv(channels[0] + channels[1], channels[2], 1, dimensions=dimensions),
            ResidualBlock(
                channels[2],
                scale_initialisation=scale_initialisation,
                dimensions=dimensions,
            ),
        )

    def forward(self, x, y):
        x = interleave(x, y)
        return self.layer(x)


class BottomUpLayer(nn.Module):
    """
    Residual blocks, each optionally downsampling.

    Args:
        n_res_blocks (int): The number of residual blocks in the layer.
        n_filters (int): The number of filters in each residual block.
        downsampling_steps (list(int)): Halvings per dimension.
        scale_initialisation (bool): For stability, scale the last layer in the residual block by 1/(depth**0.5).
            See VDVAE, Child 2020.
        dimensions (int): Dimensionality of the data (1, 2 or 3)
    """

    def __init__(
        self,
        n_res_blocks,
        n_filters,
        downsampling_steps=(0, 0, 0),
        scale_initialisation=False,
        dimensions=2,
    ):
        super().__init__()

        self.bu_blocks = nn.Sequential()
        for _ in range(n_res_blocks):
            self.bu_blocks.append(
                ResBlockWithResampling(
                    c_in=n_filters,
                    c_out=n_filters,
                    resample="down",
                    n_resample=downsampling_steps,
                    scale_initialisation=scale_initialisation,
                    dimensions=dimensions,
                )
            )

    def forward(self, x):
        return self.bu_blocks(x)


class TopDownLayer(nn.Module):
    """
    Deterministic top-down layer of the UNet: residual blocks, each optionally
    upsampling.

    Merges a bottom-up skip connection and optionally a skip connection from the previous layer.

    Args:
        n_res_blocks (int): The number of residual blocks in the layer.
        n_filters (int): The number of filters in each residual block.
        is_top_layer (bool): Whether the layer is the top layer.
        upsampling_steps (list(int)): Doublings per dimension.
        skip (bool, optional): Whether to use a skip connection from the previous layer. Defaults to False.
        scale_initialisation (bool): For stability, scale the last layer in the residual block by 1/(depth**0.5).
            See VDVAE, Child 2020.
        dimensions (int): Dimensionality of the data (1, 2 or 3)
    """

    def __init__(
        self,
        n_res_blocks,
        n_filters,
        is_top_layer=False,
        upsampling_steps=(0, 0, 0),
        skip=False,
        scale_initialisation=False,
        dimensions=2,
    ):
        super().__init__()

        self.is_top_layer = is_top_layer
        self.skip = skip

        self.blocks = nn.Sequential()
        for _ in range(n_res_blocks):
            self.blocks.append(
                ResBlockWithResampling(
                    n_filters,
                    n_filters,
                    resample="up",
                    n_resample=upsampling_steps,
                    scale_initialisation=scale_initialisation,
                    dimensions=dimensions,
                )
            )

        if not is_top_layer:
            self.merge = MergeLayer(
                channels=n_filters,
                scale_initialisation=scale_initialisation,
                dimensions=dimensions,
            )

            if skip:
                self.skip_connection_merger = MergeLayer(
                    channels=n_filters,
                    scale_initialisation=scale_initialisation,
                    dimensions=dimensions,
                )

    def forward(
        self,
        input_=None,
        skip_connection_input=None,
        bu_value=None,
    ):
        if self.is_top_layer:
            output = bu_value
        else:
            output = self.merge(bu_value, input_)

        # Skip connection from previous layer
        if self.skip and not self.is_top_layer:
            output = self.skip_connection_merger(output, skip_connection_input)

        # Last top-down block (sequence of residual blocks)
        output = self.blocks(output)

        return output


class NormalStochasticBlock(nn.Module):
    """
    Stochastic layer for the inference/generation path of a Ladder VAE.

    Creates p(z{i}|z{i+1}) (and q(z{i}|z{i+1}, x)) for generation (or inference).

    Args:
        c_in (int): Number of input channels.
        c_vars (int): Number of latent variable channels.
        c_out (int): Number of output channels.
        kernel (int): Kernel size for the convolutional layers.
        transform_p_params (bool): Whether to double the channels of p(z) with a convolutional layer.
        dimensions (int): Dimensionality of the data (1, 2 or 3)
    """

    # Added to both standard deviations so that one can't underflow to 0, which
    # makes the KL's log(std) -inf and the KL NaN. Far below any useful std.
    min_std = 1e-5

    def __init__(self, c_in, c_vars, c_out, kernel=3, transform_p_params=True, dimensions=2):
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        self.transform_p_params = transform_p_params
        self.c_in = c_in
        self.c_out = c_out
        self.c_vars = c_vars

        if transform_p_params:
            self.conv_in_p = Conv(
                c_in,
                2 * c_vars,
                kernel,
                padding=pad,
                padding_mode="reflect",
                dimensions=dimensions,
            )
        self.conv_in_q = Conv(
            c_in,
            2 * c_vars,
            kernel,
            padding=pad,
            padding_mode="reflect",
            dimensions=dimensions,
        )
        self.conv_out = Conv(
            c_vars,
            c_out,
            kernel,
            padding=pad,
            padding_mode="reflect",
            dimensions=dimensions,
        )

    def _mean_std(self, params):
        # In float32 whatever the autocast dtype: the KL sums millions of terms.
        mu, std_ = torch.chunk(params.float(), dim=1, chunks=2)
        std = nn.functional.softplus(std_) + self.min_std
        return mu, std

    def forward(self, p_params, q_params=None):
        """
        Forward pass of the stochastic layer.

        Args:
            p_params (torch.Tensor): Parameters of the prior distribution p(z).
            q_params (torch.Tensor, optional): Parameters of the approximate posterior distribution q(z|x).

        Returns:
            torch.Tensor: Sample from either q(z|x) or p(z), through `conv_out`.
            dict: mean and std of q(z|x), None without q
            dict: mean and std of p(z)
        """

        if self.transform_p_params:
            p_params = self.conv_in_p(p_params)
        else:
            assert p_params.size(1) == 2 * self.c_vars

        p_mu, p_std = self._mean_std(p_params)

        if q_params is not None:
            q_mu, q_std = self._mean_std(self.conv_in_q(q_params))
            mu, std = q_mu, q_std
        else:
            q_mu = q_std = None
            mu, std = p_mu, p_std

        z = mu + std * torch.randn_like(mu)

        # Output of stochastic layer
        z = self.conv_out(z)

        q_params = {"mu": q_mu, "std": q_std}
        p_params = {"mu": p_mu, "std": p_std}
        return z, q_params, p_params


class VAETopDownLayer(nn.Module):
    """
    Top-down layer for the generative path of a Ladder VAE.

    Merges a bottom-up skip connection for the approximate posterior
    and optionally a skip connection from the previous layer.

    Args:
        z_dim (int): Channels of the latent variable.
        n_res_blocks (int): Number of residual blocks in the layer.
        n_filters (int): Number of filters in each residual block.
        is_top_layer (bool): Whether the layer is the top layer.
        upsampling_steps (list(int)): Doublings per dimension.
        stochastic_skip (bool): Whether to include a skip connection around the stochastic layer.
        learn_top_prior (bool): Whether to learn the parameters of the top prior.
        top_prior_param_size (list(int)): Shape of the top prior parameters.
        scale_initialisation (bool): For stability, scale the last layer in the residual block by 1/(depth**0.5).
            See VDVAE, Child 2020.
        dimensions (int): Dimensionality of the data (1, 2 or 3)
    """

    def __init__(
        self,
        z_dim,
        n_res_blocks,
        n_filters,
        is_top_layer=False,
        upsampling_steps=(0, 0, 0),
        stochastic_skip=False,
        learn_top_prior=False,
        top_prior_param_size=None,
        scale_initialisation=False,
        dimensions=2,
    ):
        super().__init__()

        self.is_top_layer = is_top_layer
        self.z_dim = z_dim
        self.stochastic_skip = stochastic_skip

        if is_top_layer:
            self.top_prior_params = nn.Parameter(
                torch.zeros(top_prior_param_size), requires_grad=learn_top_prior
            )

        self.deterministic_blocks = nn.Sequential()
        for _ in range(n_res_blocks):
            self.deterministic_blocks.append(
                ResBlockWithResampling(
                    n_filters,
                    n_filters,
                    resample="up",
                    n_resample=upsampling_steps,
                    scale_initialisation=scale_initialisation,
                    dimensions=dimensions,
                )
            )

        self.stochastic = NormalStochasticBlock(
            c_in=n_filters,
            c_vars=z_dim,
            c_out=n_filters,
            transform_p_params=(not is_top_layer),
            dimensions=dimensions,
        )

        if not is_top_layer:
            self.merge = MergeLayer(
                channels=n_filters,
                scale_initialisation=scale_initialisation,
                dimensions=dimensions,
            )

            if stochastic_skip:
                self.skip_connection_merger = MergeLayer(
                    channels=n_filters,
                    scale_initialisation=scale_initialisation,
                    dimensions=dimensions,
                )

    def forward(
        self,
        p_params=None,
        skip_connection_input=None,
        bu_value=None,
        n_img_prior=None,
    ):
        """
        Forward pass of the top-down layer.

        Args:
            p_params (torch.Tensor, optional): Output of the layer above.
            skip_connection_input (torch.Tensor, optional): Skip connection from the previous layer.
            bu_value (torch.Tensor, optional): Inferred bottom-up value at this layer.
            n_img_prior (int, optional): Number of images to generate.

        Returns:
            torch.Tensor: Output of the layer.
            dict: q(z|x)
            dict: p(z)
        """

        if self.is_top_layer:
            assert p_params is None
            p_params = self.top_prior_params

            # Sample specific number of images by expanding the prior
            if n_img_prior is not None:
                p_params = p_params.repeat_interleave(n_img_prior, dim=0)

        # In inference mode, get parameters of q from inference path,
        # merging with top-down path if it's not the top layer
        if bu_value is not None:
            if self.is_top_layer:
                q_params = bu_value
            else:
                q_params = self.merge(bu_value, p_params)
        # In generative mode, q is not used
        else:
            q_params = None

        # Samples from either q(z_i | z_{i+1}, x) or p(z_i | z_{i+1})
        # depending on whether q_params is None
        z, q, p = self.stochastic(
            p_params=p_params,
            q_params=q_params,
        )

        # Skip connection from previous layer
        if self.stochastic_skip and not self.is_top_layer:
            z = self.skip_connection_merger(z, skip_connection_input)

        z = self.deterministic_blocks(z)

        return z, q, p
