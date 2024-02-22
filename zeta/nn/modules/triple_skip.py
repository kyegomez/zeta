import torch
from torch import nn


class TripleSkipBlock(nn.Module):
    def __init__(self, submodule1, submodule2, submodule3):
        """
        TripleSkipBlock class represents a block that performs triple skip connections.

        Args:
            submodule1 (nn.Module): The first submodule.
            submodule2 (nn.Module): The second submodule.
            submodule3 (nn.Module): The third submodule.
        """
        super().__init__()
        self.submodule1 = submodule1
        self.submodule2 = submodule2
        self.submodule3 = submodule3

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the TripleSkipBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying triple skip connections.
        """
        return x + self.submodule1(x + self.submodule2(x + self.submodule(x)))
