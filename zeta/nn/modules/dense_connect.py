import torch
from torch import nn


class DenseBlock(nn.Module):
    def __init__(self, submodule, *args, **kwargs):
        """
        Initializes a DenseBlock module.

        Args:
            submodule (nn.Module): The submodule to be applied in the forward pass.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DenseBlock module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the DenseBlock operation.
        """
        return torch.cat([x, self.submodule(x)], dim=1)
