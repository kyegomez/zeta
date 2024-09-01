from einops import rearrange
from torch import Tensor, nn


def audio_to_text(x: Tensor, seqlen: int, dim: int, norm: bool = True):
    """
    Reshapes and projects the audio input tensor to text representation.

    Args:
        x (Tensor): Input audio tensor of shape (batch_size, sequence_length, dim).
        seqlen (int): Length of the output sequence.
        dim (int): Dimension of the projected audio tensor.
        norm (bool, optional): Whether to apply layer normalization. Defaults to True.

    Returns:
        Tensor: Reshaped and projected audio tensor of shape (batch_size, seqlen, dim).

    Example::
        >>> x = torch.randn(2, 10, 80)
        >>> x = audio_to_text(x, 100, 512)
        >>> x.shape
        torch.Size([2, 100, 512])
    """
    audio = rearrange(x, "b l -> b l 1")

    # Audio dimensions
    b, l, d = audio.shape
    audio_proj = nn.Linear(d, dim)(audio)

    # Reshape and project the seqlen
    audio = rearrange(audio_proj, "b l d -> b d l")
    audio_proj2 = nn.Linear(l, seqlen)(audio)
    audio = rearrange(audio_proj2, "b d l -> b l d")

    if norm:
        audio = nn.LayerNorm(dim)(audio)

    return audio
