import torch
from torch import nn


class DynamicRoutingBlock(nn.Module):
    def __init__(self, sb1, sb2, routing_module):
        """
        A module that performs dynamic routing between two sub-blocks based on routing weights.

        Args:
            sb1 (nn.Module): The first sub-block.
            sb2 (nn.Module): The second sub-block.
            routing_module (nn.Module): The module that computes routing weights.

        """
        super().__init__()
        self.sb1 = sb1
        self.sb2 = sb2
        self.routing_module = routing_module

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the dynamic routing block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after dynamic routing.

        """
        routing_weights = self.routing_module(x)
        return routing_weights * self.sb1(x) + (1 - routing_weights) * self.sb2(
            x
        )
