from torch import nn, Tensor


class MonarchMLP(nn.Module):
    """
    A sparse MLP from this paper: https://hazyresearch.stanford.edu/blog/2024-01-11-m2-bert-retrieval

    Example:
        >>> x = torch.randn(1, 3, 32, 32)
        >>> model = MonarchMLP()
        >>> out = model(x)
        >>> print(out)
    """

    def __init__(
        self,
    ):
        super().__init__()

        self.glu = nn.GLU()
        self.gelu = nn.GELU()

    def forward(self, x: Tensor):
        """
        Forward pass of the MonarchMLP model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through GLU and GELU activation functions.
        """
        x = self.glu(x)
        x = self.gelu(x)
        return x
