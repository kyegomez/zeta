import torch
from torch import nn


class Scale(nn.Module):
    """
    Scale

    Args:
        value (float): scale value
        fn (callable): function to scale


    Attributes:
        value (float): scale value
        fn (callable): function to scale


    Usage:
        >>> x = torch.randn(1, 10, 512)
        >>> scale = Scale(value=0.5, fn=torch.sin)
        >>> scale(x).shape
        torch.Size([1, 10, 512])

    """

    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        """Forward method of Scale"""
        out = self.fn(x, **kwargs)

        def scale_fn(t):
            return t * self.value

        if not isinstance(out, tuple):
            return scale_fn(out)

        return (scale_fn(out[0]), *out[1:])
