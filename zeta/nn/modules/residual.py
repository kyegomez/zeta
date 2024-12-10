from torch import nn


class Residual(nn.Module):
    """
    Residual connection.

    Args:
        fn (nn.Module): The module.

    Example:
        >>> module = Residual(nn.Linear(10, 10))
        >>> x = torch.randn(10, 10)
        >>> y = module(x)
        >>> y.shape
        torch.Size([10, 10])

    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        """Forward method implementation."""
        return self.fn(x) + x
