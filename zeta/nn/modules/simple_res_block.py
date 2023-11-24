import torch
from torch import nn


class SimpleResBlock(nn.Module):
    """
    Simple residual block with GELU activation

    Args:
        channels: number of input/output channels

    Returns:
        x + proj(x) where proj is a small MLP

    Usage:
        >>> block = SimpleResBlock(256)
        >>> x = torch.randn(4, 256)
        >>> block(x).shape
        torch.Size([4, 256])

    """

    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        """Forward pass"""
        x = self.pre_norm(x)
        return x + self.proj(x)
