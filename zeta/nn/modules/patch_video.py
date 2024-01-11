from einops import rearrange


def patch_video(x, patch_size: int):
    """
    Patch a video into patches of size patch_size x patch_size x patch_size x C x H x W

    Args:
        x (torch.Tensor): Input video tensor of shape (batch_size, time, channels, height, width).
        patch_size (int): Size of the patches in each dimension.

    Returns:
        torch.Tensor: Patched video tensor of shape (batch_size, time, height, width, patch_size, patch_size, patch_size, channels).

    Example::
        >>> x = torch.randn(2, 10, 3, 32, 32)
        >>> x = patch_video(x, 4)
        >>> x.shape
        torch.Size([2, 10, 8, 8, 4, 4, 4, 3])
    """
    b, t, c, h, w = x.shape
    x = rearrange(
        x, "b t c h w -> b c t h w"
    )  # change shape to (batch_size, channels, time, height, width)
    x = rearrange(
        x,
        "b c (t p1) (h p2) (w p3) -> b t h w (p1 p2 p3) c",
        p1=patch_size,
        p2=patch_size,
        p3=patch_size,
    )
    return x
