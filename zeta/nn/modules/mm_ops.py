from einops import rearrange, reduce
from torch import Tensor, nn


def threed_to_text(
    x: Tensor, max_seq_len: int, dim: int, flatten: bool = False
):
    """
    Converts a 3D tensor to text representation.

    Args:
        x (Tensor): The input tensor of shape (batch_size, sequence_length, input_dim).
        max_seq_len (int): The maximum sequence length of the output tensor.
        dim (int): The dimension of the intermediate tensor.
        flatten (bool, optional): Whether to flatten the intermediate tensor. Defaults to False.

    Returns:
        Tensor: The output tensor of shape (batch_size, max_seq_len, input_dim).
    """
    b, s, d = x.shape

    x = nn.Linear(d, dim)(x)

    x = rearrange(x, "b s d -> b d s")
    x = nn.Linear(s, max_seq_len)(x)
    x = rearrange(x, "b d s -> b s d")
    return x


def text_to_twod(x: Tensor, dim: int):
    """
    Converts a 3D tensor of shape (batch_size, sequence_length, input_dim) to a 2D tensor of shape (batch_size, dim)
    by averaging the sequence dimension and applying a linear transformation.

    Args:
        x (Tensor): The input tensor of shape (batch_size, sequence_length, input_dim).
        dim (int): The output dimension.

    Returns:
        Tensor: The output tensor of shape (batch_size, dim).
    """
    b, s, d = x.shape
    x = reduce(x, "b s d -> b d", "mean")
    x = nn.Linear(d, dim)(x)
    return x
