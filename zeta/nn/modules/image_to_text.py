from einops import rearrange, reduce
from torch import Tensor, nn


def img_to_text(x: Tensor, seqlen: int, dim: int, norm: bool = True):
    """
    Convert an image tensor to a text tensor.

    Args:
        x (Tensor): Input image tensor of shape (batch_size, channels, height, width).
        seqlen (int): Length of the output text sequence.
        dim (int): Dimension of the intermediate representation.
        norm (bool, optional): Whether to apply layer normalization. Defaults to True.

    Returns:
        Tensor: Output text tensor of shape (batch_size, seqlen, dim).

    Example::
        >>> x = torch.randn(2, 3, 32, 32)
        >>> x = img_to_text(x, 100, 512)
        >>> x.shape
        torch.Size([2, 100, 512])
    """
    b, c, h, w = x.shape

    img = reduce(x, "b c h w -> b c (h w)", "mean")
    img = nn.Linear(h * w, dim)(img)
    img = rearrange(img, "b c d -> b d c")
    img = nn.Linear(c, seqlen)(img)
    img = rearrange(img, "b d c -> b c d")

    if norm:
        img = nn.LayerNorm(dim)(img)

    return img
