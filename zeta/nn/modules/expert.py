import torch
from torch import nn


class Experts(nn.Module):
    """
    Expert module for the Mixture of Experts layer.

    Args:
        dim (int): Dimension of the input features.
        experts (int): Number of experts.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).

    Examples:
        >>> x = torch.randn(1, 3, 512)
        >>> model = Expert(512, 16)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([1, 3, 512])

    """

    def __init__(
        self,
        dim: int,
        experts: int = 16,
    ):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(experts, dim, dim * 2))
        self.w2 = nn.Parameter(torch.randn(experts, dim * 4, dim * 4))
        self.w3 = nn.Parameter(torch.randn(experts, dim * 4, dim))
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        """Forward pass."""
        hidden1 = self.act(torch.einsum("end,edh->enh", x, self.w1))
        hidden2 = self.act(torch.einsum("end,edh->enh", hidden1, self.w2))
        out = torch.einsum("end,edh->enh", hidden2, self.w3)
        return out
