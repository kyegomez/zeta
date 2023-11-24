import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


def exists(val):
    return val is not None


def shift(t, amount, mask=None):
    if amount == 0:
        return t
    else:
        amount = min(amount, t.shape[-1])

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.0)

    return pad_at_dim(t, (amount, -amount), dim=-2, value=0.0)


class ShiftTokens(nn.Module):
    """
    Shift Tokens

    Overview: Shift tokens in the input sequence

    Args:
        shifts (list): list of shifts
        fn (nn.Module): function to apply after shifting

    Attributes:
        shifts (tuple): tuple of shifts
        fn (nn.Module): function to apply after shifting

    Usage:
        >>> x = torch.randn(1, 10, 512)
        >>> shift_tokens = ShiftTokens(shifts=[1, 2, 3], fn=nn.Linear(512, 512))
        >>> shift_tokens(x).shape
        torch.Size([1, 10, 512])

    """

    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        """Forward method of ShiftTokens"""
        mask = kwargs.get("mask", None)
        shifts = self.shifts
        segments = len(shifts)

        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim=-1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(
            map(
                lambda args: shift(*args, mask=mask),
                zip(segments_to_shift, shifts),
            )
        )
        x = torch.cat((*segments_to_shift, *rest), dim=-1)
        return self.fn(x, **kwargs)
