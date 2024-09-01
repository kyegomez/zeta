import torch
from torch import nn


class PositionInterpolationEmbeddings(nn.Module):
    """
    PositionalEmbedding module that uses interpolation to generate positional embeddings.

    Args:
        dim (int, optional): Dimension of the model. Defaults to None.
        max_positions (int, optional): Maximum length of the input sequence. Defaults to 2048.
        base (int, optional): Base value. Defaults to 10000.
        device ([type], optional): Device to use. Defaults to None.

    Example:
        >>> positional_embedding = PositionInterpolationEmbeddings(512, 1000)
        >>> x = torch.randn(32, 100, 512)
        >>> positions = torch.arange(100)
        >>> embedded_tensor = positional_embedding(x, positions)

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
