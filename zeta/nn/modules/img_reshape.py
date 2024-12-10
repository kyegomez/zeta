from einops import rearrange


def image_reshape(img_batch):
    """
    Reshapes an image batch

    Einstein summation notation:
        'b' = batch size
        'h' = image height
        'w' = image width
        'c' = number of channels

    Args:
        img_batch (torch.Tensor): Image batch to be reshaped

    Returns:
        torch.Tensor: Reshaped image batch

    Usage:
        >>> img_batch = torch.rand(1, 3, 224, 224)
        >>> img_batch.shape
        torch.Size([1, 3, 224, 224])
        >>> img_batch = image_reshape(img_batch)
        >>> img_batch.shape
        torch.Size([1, 224, 224, 3])

    """
    return rearrange(img_batch, "b h w c -> b c h w")


# #random
# x = torch.rand(1, 3, 224, 224)
# model = image_reshape(x)
# print(model)
