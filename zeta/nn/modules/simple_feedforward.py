from torch import nn


def SimpleFeedForward(dim: int, hidden_dim: int, dropout=0.1):
    """
    Feedforward neural network with LayerNorms and GELU activations


    Flow:
    layer_norm -> linear -> gelu -> linear -> dropout


    Args:
        dim (int): Input dimension
        hidden_dim (int): Hidden dimension
        dropout (float): Dropout probability

    Usage:
    >>> model = SimpleFeedForward(768, 2048, 0.1)
    >>> x = torch.randn(1, 768)
    >>> model(x).shape
    torch.Size([1, 768])
    """
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )
