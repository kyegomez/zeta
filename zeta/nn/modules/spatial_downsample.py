import torch
from torch import nn
from einops import rearrange, pack, unpack

# utils
# helper


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def identity(t):
    return t


def divisible_by(num, den):
    return (num % den) == 0


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def is_odd(n):
    return not divisible_by(n, 2)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


class SpatialDownsample(nn.Module):
    """
    Spatial Downsample Module
    -------------------------

    This module is used to downsample the spatial dimension of a tensor.
    It is used in the ResNet architecture to downsample the spatial dimension
    of the input tensor by a factor of 2.

    Args:
        dim (int): The number of channels in the input tensor.
        dim_out (int): The number of channels in the output tensor.
        kernel_size (int): The size of the kernel used to downsample the input tensor.

    Returns:
        out (tensor): The downsampled tensor.

    Usage:
        >>> downsample = SpatialDownsample(64, 128, 3)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> out = downsample(x)
        >>> out.shape
        torch.Size([1, 128, 16, 16])

    """

    def __init__(
        self,
        dim,
        dim_out=None,
        kernel_size=3,
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.conv = nn.Conv3d(
            dim,
            dim_out,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        """
        Forward pass of the SpatialDownsample module.

        """
        x = rearrange(x, "b c t h w -> b t c h w")
        x, ps = pack_one(x, "* c h w")

        out = self.conv(x)

        out = unpack_one(out, ps, "* c h w")
        out = rearrange(out, "b t c h w -> b c t h w")
        return out
