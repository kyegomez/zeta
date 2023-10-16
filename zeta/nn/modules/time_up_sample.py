import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange, pack, unpack

from zeta.utils.main import default

# helper


def exists(v):
    return v is not None


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


class TimeUpSample2x(nn.Module):
    """
    Time Up Sample Module

    This module is used to upsample the time dimension of a tensor.

    Args:
        dim (int): The number of channels in the input tensor.
        dim_out (int): The number of channels in the output tensor.

    Returns:
        out (tensor): The upsampled tensor.

    Usage:
        >>> upsample = TimeUpSample2x(64, 128)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> out = upsample(x)
        >>> out.shape
        torch.Size([1, 128, 32, 32])

    """

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv1d(dim, dim_out * 2, 1)

        self.net = nn.Sequential(
            conv, nn.SiLU(), Rearrange("b (c p) t -> b c (t p)", p=2)
        )

        self.init_conv(conv)

    def init_conv(self, conv):
        """iNIITIALIZE CONVOLUTIONAL LAYER"""
        o, i, t = conv.weight.shape
        conv_weight = torch.empty(0 // 2, i, t)
        nn.init.kaiming_normal_(conv_weight)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        """
        Einstein notation

        b - batch
        c - channels
        t - time
        h - height
        w - width


        """
        x = rearrange(x, "b c t h w -> b h w c t")
        x, ps = pack_one(x, "* c t")

        out = self.net(x)

        out = unpack_one(out, ps, "* c t")
        out = rearrange(out, "b h w c t -> b c t h w")
        return out
