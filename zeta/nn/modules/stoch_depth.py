import torch
from torch import nn


class StochDepth(nn.Module):
    def __init__(self, stochdepth_rate: float):
        """
        Initializes a Stochastic Depth module.

        Args:
            stochdepth_rate (float): The probability of dropping each input activation.
        """
        super().__init__()
        self.stochdepth_rate = stochdepth_rate

    def forward(self, x):
        """
        Forward pass of the Stochastic Depth module.

        Args:
            x: The input tensor.

        Returns:
            The output tensor after applying stochastic depth.
        """
        if not self.training:
            return x

        batch_size = x.shape[0]
        rand_tensor = torch.rand(
            batch_size,
            1,
            1,
            1,
        ).type_as(x)
        keep_prob = 1 - self.stochdepth_rate
        binary_tensor = torch.floor(rand_tensor + keep_prob)

        return x * binary_tensor
