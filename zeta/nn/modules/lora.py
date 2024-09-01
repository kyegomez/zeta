import torch
from torch import nn


class Lora(nn.Module):
    """
    Lora module applies a linear transformation to the input tensor using the Lora algorithm.

    Args:
        dim (int): The input dimension.
        dim_out (int): The output dimension.
        r (int, optional): The rank of the transformation. Defaults to 8.
        alpha (float, optional): The scaling factor. Defaults to None.

    Attributes:
        scale (float): The scaling factor calculated as alpha / r.
        A (nn.Parameter): The learnable parameter representing the input-to-hidden transformation matrix.
        B (nn.Parameter): The learnable parameter representing the hidden-to-output transformation matrix.

    Properties:
        weight (torch.Tensor): The weight matrix obtained by multiplying A and B and scaling it by the scale factor.

    Methods:
        forward(x): Applies the Lora transformation to the input tensor x.

    """

    def __init__(self, dim: int, dim_out: int, r: int = 8, alpha: float = 2):
        super().__init__()
        self.scale: float = alpha / r

        self.A: nn.Parameter = nn.Parameter(torch.randn(dim, r))
        self.B: nn.Parameter = nn.Parameter(torch.randn(r, dim_out))

    @property
    def weight(self) -> torch.Tensor:
        """Weight matrix obtained by multiplying A and B and scaling it by the scale factor.

        Returns:
            torch.Tensor: The weight matrix.
        """
        return (self.A @ self.B) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Lora module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return x @ self.weight
