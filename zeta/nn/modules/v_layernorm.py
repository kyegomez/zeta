import torch
from torch import Tensor, nn


class VLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Initializes a VLayerNorm module.

        Args:
            dim (int): The input dimension.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-5.
        """
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: Tensor):
        """
        Performs a forward pass of the VLayerNorm module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The normalized tensor after applying VLayerNorm.
        """
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b
