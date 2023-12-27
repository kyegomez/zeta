import torch
from torch import nn


class StochasticSkipBlocK(nn.Module):
    """
    A module that implements stochastic skip connections in a neural network.

    Args:
        sb1 (nn.Module): The module to be skipped with a certain probability.
        p (float): The probability of skipping the module. Default is 0.5.

    Returns:
        torch.Tensor: The output tensor after applying the stochastic skip connection.
    """

    def __init__(self, sb1, p=0.5):
        super().__init__()
        self.sb1 = sb1
        self.p = p

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the StochasticDepth module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the StochasticDepth module.
        """
        if self.training and torch.rand(1).item() < self.p:
            return x  # Skip the sb1
        else:
            return self.sb1(x)
