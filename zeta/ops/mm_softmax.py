from torch import Tensor
import torch.nn.functional as F


def mm_softmax(
    x: Tensor,
    y: Tensor,
    weight: float = 1.0,
    weight2: float = 1.0,
    temp: float = 1.0,
):
    """
    Applies softmax function to the element-wise product of two input tensors, x and y.

    Args:
        x (Tensor): The first input tensor.
        y (Tensor): The second input tensor.
        weight (float, optional): Weight multiplier for x. Defaults to 1.0.
        weight2 (float, optional): Weight multiplier for y. Defaults to 1.0.
        temp (float, optional): Temperature scaling factor. Defaults to 1.0.

    Returns:
        Tensor: The softmax output tensor.
    """
    assert x.size() == y.size(), "x and y must have the same shape"

    # Combine modalities
    combined_data = weight * x * weight2 * y

    # Apply temperature scaling
    scaled_data = combined_data / temp

    # Compute softmax on scaled combined data
    softmax = F.softmax(scaled_data, dim=-1)

    return softmax
