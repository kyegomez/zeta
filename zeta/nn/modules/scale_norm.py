import torch
from torch import nn, Tensor


class ScaleNorm(nn.Module):
    """
    Applies scale normalization to the input tensor along the last dimension.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-5.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = eps

        self.g = nn.Parameter(torch.ones(1) * (dim**-0.5))

    def forward(self, x: Tensor):
        """
        Applies scale normalization to the input tensor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The scale-normalized tensor.
        """
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x / norm.clamp(min=self.eps) + self.g
