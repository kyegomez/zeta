import torch
from torch import nn


class GatedResidualBlock(nn.Module):
    def __init__(self, sb1, gate_module):
        """
        Gated Residual Block module.

        Args:
            sb1 (nn.Module): The first sub-block.
            gate_module (nn.Module): The gate module.

        """
        super().__init__()
        self.sb1 = sb1
        self.gate_module = gate_module

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the Gated Residual Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        gate = torch.sigmoid(self.gate_module(x))
        return x + gate * self.sb1(x)
