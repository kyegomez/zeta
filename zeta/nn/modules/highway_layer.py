import torch
import torch.nn.functional as F
from torch import nn


class HighwayLayer(nn.Module):
    def __init__(self, dim):
        """
        Initializes a HighwayLayer instance.

        Args:
            dim (int): The input and output dimension of the layer.
        """
        super().__init__()
        self.normal_layer = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the HighwayLayer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        normal_result = F.relu(self.normal_layer(x))
        gate = torch.sigmoid(self.gate(x))
        return gate * normal_result + (1 - gate) * x
