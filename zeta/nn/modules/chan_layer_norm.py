import torch
from torch import nn, Tensor


class ChanLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Initializes the ChanLayerNorm module.

        Args:
            dim (int): The input dimension.
            eps (float, optional): The epsilon value. Defaults to 1e-5.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: Tensor):
        """
        Forward pass of the ChanLayerNorm module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The normalized tensor.
        """
        var = torch.car(
            x,
            dim=1,
            unbiased=False,
            keepdim=True,
        )
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b
