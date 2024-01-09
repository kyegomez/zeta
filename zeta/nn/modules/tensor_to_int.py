import torch 
from torch import Tensor

def tensor_to_int(tensor: Tensor, reduction="sum"):
    """
    Converts a tensor to an integer value based on the specified reduction operation.

    Args:
        tensor (Tensor): The input tensor.
        reduction (str, optional): The reduction operation to be applied. 
            Valid options are "sum", "mean", and "max". Defaults to "sum".

    Returns:
        int: The integer value obtained after applying the reduction operation to the tensor.
    
    Raises:
        ValueError: If an invalid reduction operation is specified.
    """
    if reduction == "sum":
        value = tensor.sum()
    elif reduction == "mean":
        value = tensor.mean()
    elif reduction == "max":
        value = tensor.max()
    else:
        raise ValueError("Invalid reduction op. Choose from sum, mean, max.")
    
    return int(value.item())

