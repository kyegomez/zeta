import torch
import torch.nn as nn


def fixed_pos_embedding(x):
    """
    Generates fixed positional embeddings for the input tensor.

    Args:
    - x: Input tensor of shape (seq_len, dim)

    Returns:
    - sin: Sine positional embeddings of shape (seq_len, dim)
    - cos: Cosine positional embeddings of shape (seq_len, dim)
    """
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = torch.einsum(
        "i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq
    ).to(x)
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    """
    Rearranges the elements of the input tensor by rotating every two elements.

    Args:
    - x: Input tensor of shape (batch_size, seq_len, dim)

    Returns:
    - x: Rearranged tensor of shape (batch_size, seq_len, dim)
    """
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def duplicate_interleave(m):
    """
    Duplicates a matrix while interleaving the copy.

    Args:
    - m: Input matrix

    Returns:
    - m: Duplicated and interleaved matrix
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)
    m = m.repeat(1, 2)
    m = m.view(dim0, -1)
    return m


def apply_rotary_pos_emb(x, sin, cos, scale=1):
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
    - x: Input tensor of shape (batch_size, seq_len, dim)
    - sin: Sine positional embeddings of shape (seq_len, dim)
    - cos: Cosine positional embeddings of shape (seq_len, dim)
    - scale: Scaling factor for the positional embeddings

    Returns:
    - x: Tensor with applied rotary positional embeddings
    """
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    return (x * cos) + (rotate_every_two(x) * sin)


class XPOS(nn.Module):
    def __init__(self, head_dim: int = None, scale_base: int = 512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale",
            (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim),
        )

    def forward(self, x, offset=0, downscale=False):
        """
        Forward pass of the XPOS module.

        Args:
        - x: Input tensor of shape (batch_size, seq_len, dim)
        - offset: Offset value for positional embeddings
        - downscale: Boolean indicating whether to downscale the positional embeddings

        Returns:
        - x: Tensor with applied rotary positional embeddings
        """
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = (
            self.scale
            ** torch.arange(min_pos, max_pos, 1)
            .to(self.scale)
            .div(self.scale_base)[:, None]
        )
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x
