import torch
from torch import nn


class RecursiveBlock(nn.Module):
    def __init__(self, modules, iters, *args, **kwargs):
        """
        Initializes a RecursiveBlock module.

        Args:
            modules (nn.Module): The module to be applied recursively.
            iters (int): The number of iterations to apply the module.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.modules = modules
        self.iters = iters

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the RecursiveBlock module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the module recursively.
        """
        for _ in range(self.iters):
            x = self.modules(x)
        return x
