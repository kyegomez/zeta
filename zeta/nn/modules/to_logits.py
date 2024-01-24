from torch import nn

def to_logits(x, dim: int, num_tokens: int):
    """
    Converts the input tensor `x` into logits using a sequential layer.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int): The dimension along which to apply the layer normalization.
        num_tokens (int): The number of output tokens.

    Returns:
        torch.Tensor: The logits tensor.
        
    Example:
    >>> x = torch.randn(1, 10, 10)
    >>> model = to_logits(x, 10, 10)
    >>> print(model)

    """
    layer = nn.Sequential(
        nn.Softmax(-1),
        nn.LayerNorm(dim),
        nn.Linear(dim, num_tokens)
    )
    return layer(x)
