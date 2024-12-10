from einops import rearrange, reduce
from torch import Tensor, nn


def video_to_text(x: Tensor, seqlen: int, dim: int, norm: bool = True):
    """
    Convert a video tensor to a text tensor.

    Args:
        x (Tensor): Input video tensor of shape (batch_size, time, channels, height, width).
        seqlen (int): Length of the output text sequence.
        dim (int): Dimension of the intermediate representation.
        norm (bool, optional): Whether to apply layer normalization. Defaults to True.

    Returns:
        Tensor: Output text tensor of shape (batch_size, seqlen, dim).

    Example::
        >>> x = torch.randn(2, 10, 3, 32, 32)
        >>> x = video_to_text(x, 100, 512)
        >>> x.shape
        torch.Size([2, 100, 512])
    """
    b, t, c, h, w = x.shape

    x = rearrange(x, "b t c h w -> b t c (h w)")
    x = reduce(x, "b t c (h w) -> b t c", "mean", h=h, w=w)
    x = nn.Linear(c, dim)(x)
    x = rearrange(x, "b t d -> b d t")
    x = nn.Linear(t, seqlen)(x)
    x = rearrange(x, "b d t -> b t d")
    return nn.LayerNorm(dim)(x)
