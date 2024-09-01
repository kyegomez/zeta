import torch
from torch import Tensor


def absmax(x: Tensor):
    """
    Compute the absolute maximum value of a tensor.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The absolute maximum value of the tensor.
    """
    return torch.max(torch.abs(x))
