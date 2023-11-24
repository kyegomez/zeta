import torch
from torch import nn


class PositionInterpolationEmbeddings(nn.Module):
    """
    PositionInterpolation
    Overview
    ========
    Positional embeddings that interpolate between sinusoidal and learned embeddings.

    Parameters
    ==========
    dim: int
        Dimension of the input embedding.
    max_positions: int
        Maximum number of positions to embed.
    base: int
        Base of the sinusoidal embedding.
    device: torch.device
        Device to store the embeddings on.

    Attributes
    ==========
    inv_freq: torch.Tensor
        Cached inverse frequencies.
    max_seq_len_cached: int
        Maximum sequence length cached.
    scale: float
        Scale of the sinusoidal embedding.
    cos_cached: torch.Tensor
        Cached cosine values.
    sin_cached: torch.Tensor
        Cached sine values.

    Methods
    =======
    forward(x, seq_len=None)
        Forward pass of the PositionInterpolationEmbeddings.


    """

    def __init__(
        self,
        dim: int = None,
        max_positions: int = 2048,
        base: int = 10000,
        device=None,
    ):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float().to(device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        max_pos_embeds = 8192

        # build here => jit trace
        self.max_seq_len_cached = max_pos_embeds
        t = torch.arange(
            self.max_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )

        self.scale = 1 / 4
        t *= self.scale

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(
                self.max_seq_len_cached,
                device=x.device,
                dtype=self.inv_freq.dtype,
            )

            t *= self.scale
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)

            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer(
                "cos_cached", emb.cos()[None, None, :, :], persistent=False
            )
            self.register_buffer(
                "sin_cached", emb.sin()[None, None, :, :], persistent=False
            )

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )
