import torch
from torch import nn
from typing import Union, Tuple
import torch.nn.functional as F
from einops import rearrange, reduce, repeat, pack, unpack

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


# helper classes


class CausalConv3d(nn.Module):
    """
    Causal Convolution Module
    -------------------------

    This module is used to perform a causal convolution on a 3D tensor.
    It is used in the ResNet architecture to perform a causal convolution
    on the input tensor.

    Args:
        chan_in (int): The number of channels in the input tensor.
        chan_out (int): The number of channels in the output tensor.
        kernel_size (int): The size of the kernel used to perform the convolution.
        pad_mode (str): The padding mode used to pad the input tensor.
        kwargs (dict): Additional arguments to be passed to the convolution layer.

    Returns:
        out (tensor): The output tensor.

    Usage:
    >>> causal_conv = CausalConv3d(64, 128, 3)
    >>> x = torch.randn(1, 64, 32, 32)
    >>> out = causal_conv(x)



    """

    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        pad_mode="reflect",
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(
            width_kernel_size
        ), "Height and width kernel sizes must be odd"

        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stide", 2)

        self.pad_mode = pad_mode
        self.time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_causal_padding = (
            width_pad,
            width_pad,
            height_pad,
            height_pad,
            self.time_pad,
            0,
        )

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(
            chan_in,
            chan_out,
            kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs,
        )

    def forward(self, x):
        """Forward pass of the module"""
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)
