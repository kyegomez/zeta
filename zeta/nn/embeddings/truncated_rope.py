# from paper:: https://arxiv.org/pdf/2308.10882.pdf

import torch
from torch import nn


class TruncatedRotaryEmbedding(nn.Module):
    """
    Truncated rotary embeddings.

    Args:
        dim (int): The dimension of the embeddings.
        a (float): The lower bound for the truncation.
        b (float): The upper bound for the truncation.
        rho (float): The value to replace the truncated values with.

    Attributes:
        inv_freq (torch.Tensor): The inverse frequencies.
        scale (torch.Tensor): The scale.


    Example:
        >>> module = TruncatedRotaryEmbedding(10, 0.5, 1.0, 0.0)
        >>> x = torch.randn(10, 10)
        >>> y = module(x)
        >>> y.shape
        torch.Size([10, 10, 10])

    """

    def __init__(self, dim, a, b, rho):
        super().__init__()
        self.dim = dim
        self.a = a
        self.b = b
        self.rho = rho
        self.base = 10000
        self.inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", self.inv_freq)

    def forward(self, seq_len, device):
        """Forward"""
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i, j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        theta = self.base ** (
            -2 * torch.arange(0, self.dim, 2).float() / self.dim
        )
        theta_star = torch.where(
            theta >= self.b,
            theta,
            torch.where(
                theta < self.a,
                torch.zeros_like(theta),
                self.rho * torch.ones_like(theta),
            ),
        )
        theta_star = torch.cat((theta_star, theta_star), dim=-1)

        result = freqs * theta_star
        return result
