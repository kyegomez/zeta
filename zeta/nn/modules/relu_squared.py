from torch import nn
import torch.nn.functional as F


class ReluSquared(nn.Module):
    """
    Applies the ReLU activation function and squares the output.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying ReLU and squaring the result.
    """

    def forward(self, x):
        return F.relu(x) ** 2
