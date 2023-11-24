import torch
from torch import nn
from einops import rearrange


def exists(val):
    return val is not None


class Residual(nn.Module):
    def __init__(self, dim, scale_residual=False, scale_residual_constant=1.0):
        super().__init__()
        self.residual_scale = (
            nn.Parameter(torch.ones(dim)) if scale_residual else None
        )
        self.scale_residual_constant = scale_residual_constant

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual


class GRUGating(nn.Module):
    """
    GRUGating
    Overview: GRU gating mechanism

    Args:
        dim (int): dimension of the embedding
        scale_residual (bool): whether to scale residual
        kwargs (dict): keyword arguments for nn.GRUCell

    Atrributes:
        gru (nn.GRUCell): GRU cell
        residual_scale (nn.Parameter): residual scale

    Usage:
        >>> x = torch.randn(1, 10, 512)
        >>> gru_gating = GRUGating(dim=512)
        >>> gru_gating(x, x).shape
        torch.Size([1, 10, 512])

    """

    def __init__(self, dim, scale_residual=False, **kwargs):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.residual_scale = (
            nn.Parameter(torch.ones(dim)) if scale_residual else None
        )

    def forward(self, x, residual):
        """Forward method of GRUGating"""
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        gated_output = self.gru(
            rearrange(x, "b n d -> (b n) d"),
            rearrange(residual, "b n d -> (b n) d"),
        )

        return gated_output.reshape_as(x)
