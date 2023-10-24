import torch
from torch import nn, einsum


def exists(val):
    return val is not None


def divisible_by(a, b):
    return a % b == 0


class ScaledSinuosidalEmbedding(nn.Module):
    """
    scaled sinusoidal embedding

    Args:
        dim (int): dimension of the embedding

    Returns:
        torch.Tensor: embedding of shape (seq_len, dim)


    Usage:
        >>> embed = ScaledSinuosidalEmbedding(dim=512)
        >>> x = torch.randn(1, 1024, 512)
        >>> pos = torch.randint(0, 1024, (1, 1024))
        >>> embed(x, pos).shape
        torch.Size([1, 1024, 512])


    """

    def __init__(self, dim: int, theta=10000):
        super().__init__()
        assert divisible_by(dim, 2)

        self.scale = nn.Parameter(torch.ones(1) * dim**-0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta**-freq_seq
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, pos=None, seq_start_pos=None):
        """Forward method of scaled sinusoidal embedding"""
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = torch.arange(seq_len, device=device)

        if exists(seq_start_pos):
            pos = pos - seq_start_pos[..., None]

        emb = einsum("i, j -> i j", pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale
