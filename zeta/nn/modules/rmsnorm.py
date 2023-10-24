import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """
    RMSNorm

    Args:
        dim (int): dimension of the embedding


    Attributes:
        g (nn.Parameter): scaling parameter
        eps (float): epsilon value

    Usage:
    We can use RMSNorm as a layer in a neural network as follows:
        >>> x = torch.randn(1, 10, 512)
        >>> rms_norm = RMSNorm(dim=512)
        >>> rms_norm(x).shape
        torch.Size([1, 10, 512])


    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**-0.5

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g
