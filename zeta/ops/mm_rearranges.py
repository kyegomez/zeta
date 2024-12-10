from einops import rearrange
from torch import Tensor


def reshape_img_to_text(x: Tensor):
    """
    Reshapes the image tensor to the same size as the text tensor.
    From B, C, H, W to B, Seqlen, Dimension using rearrange.

    Args:
        x (Tensor): The image tensor.

    Returns:
        Tensor: The reshaped image tensor.

    """
    b, c, h, w = x.shape
    out = rearrange(x, "b c h w -> b (h w) c")
    return out


def reshape_text_to_img(x: Tensor, h: int, w: int):
    """
    Reshapes the text tensor to the same size as the image tensor.
    From B, Seqlen, Dimension to B, C, H, W using rearrange.

    Args:
        x (Tensor): The text tensor.
        h (int): The height of the image.
        w (int): The width of the image.

    Returns:
        Tensor: The reshaped text tensor.

    """
    b, seqlen, dim = x.shape
    out = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
    return out


def reshape_video_to_text(x: Tensor):
    """
    Reshapes the video tensor to the same size as the text tensor.
    From B, C, T, H, W to B, Seqlen, Dimension using rearrange.

    Args:
        x (Tensor): The video tensor.

    Returns:
        Tensor: The reshaped video tensor.

    """
    b, c, t, h, w = x.shape
    out = rearrange(x, "b c t h w -> b (t h w) c")
    return out


def reshape_audio_to_text(x: Tensor):
    """
    Reshapes the audio tensor to the same size as the text tensor.
    From B, C, T to B, Seqlen, Dimension using rearrange.

    Args:
        x (Tensor): The audio tensor.

    Returns:
        Tensor: The reshaped audio tensor.

    """
    b, c, t = x.shape
    out = rearrange(x, "b c t -> b t c")
    return out
