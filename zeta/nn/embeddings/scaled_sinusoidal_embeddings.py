import torch
from torch import nn, Tensor, einsum

from zeta.utils.main import divisible_by


class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, theta: int = 10000):
        """
        Initializes a ScaledSinusoidalEmbedding module.

        Args:
            dim (int): The dimension of the embedding.
            theta (int, optional): The scaling factor for the sinusoidal frequencies. Defaults to 10000.
        """
        super().__init__()
        assert divisible_by(dim, 2)
        self.scale = nn.Parameter(torch.ones(1) * dim**-0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta**-freq_seq
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor, pos=None, seq_start_pos=None):
        """
        Forward pass of the ScaledSinusoidalEmbedding module.

        Args:
            x (Tensor): The input tensor.
            pos (Tensor, optional): The position tensor. Defaults to None.
            seq_start_pos (Tensor, optional): The starting position tensor for sequences. Defaults to None.

        Returns:
            Tensor: The embedded tensor.
        """
        sq, device = x.shape[1], x.device

        if pos is not None:
            pos = torch.arange(sq, device=device)

        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]

        emb = einsum("i, j -> i j", pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale
