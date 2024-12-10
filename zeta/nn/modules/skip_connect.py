import torch
from torch import nn


class SkipConnection(nn.Module):
    def __init__(self, submodule):
        super().__init__()
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SkipConnection module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after adding the input tensor with the submodule output.
        """
        return x + self.submodule(x)
