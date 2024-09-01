import torch
from torch import nn


class ScaleNorm(nn.Module):
    """
    ScaleNorm

    Args:
        dim (int): dimension of the embedding
        eps (float): epsilon value

    Attributes:
        g (nn.Parameter): scaling parameter

    Usage:
    We can use ScaleNorm as a layer in a neural network as follows:
        >>> x = torch.randn(1, 10, 512)
        >>> scale_norm = ScaleNorm(dim=512)
        >>> scale_norm(x).shape
        torch.Size([1, 10, 512])

    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1) * (dim**-0.5))

    def forward(self, x):
        """Forward method of ScaleNorm"""
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x / norm.clamp(min=self.eps) * self.g
