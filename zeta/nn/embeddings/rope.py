# from paper:: https://arxiv.org/pdf/2308.10882.pdf

import torch
from torch import nn
from einops import rearrange


def exists(val):
    return val is not None


class RotaryEmbedding(nn.Module):
    """
    Rotary Embeddings

    Args:

    # Future improvements
    - add z value, how far along is sequence is particular embedding, => Z access would be giving a way to rotary embeddings
    tell the difference between the first instance of a phrase and the last phrase of a doc,

    """

    def __init__(
        self,
        dim,
        use_xpos=False,
        scale_base=512,
        interpolation_factor=1.0,
        base=10000,  # 500,000 for long content
        base_rescale_factor=1.0,
    ):
        super().__init__()
        # rscal rotary embeddings to long sequence length without finetuning
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        assert interpolation_factor >= 1
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer("scale", None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = scale_base
        self.register_buffer("scale", scale)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        t = t / self.interpolation_factor

        freqs = torch.einsum("i, j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        if not exists(self.scale):
            return freqs, 1.0

        power = (
            torch.arange(seq_len, device=device) - (seq_len // 2)
        ) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = torch.cat((scale, scale), dim=-1)

        return freqs, scale


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs, scale=1):
    seq_len = t.shape[-2]
    freqs = freqs[-seq_len:, :]
    return (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
