import torch
from einops import rearrange


def expand_channels(tensor, factor: int = 2):
    """
    Expand the channel dimenions of a tensor

    Einstein summation notation:
        'b' = batch size
        'c' = number of channels
        '...' = spatial dimensions

    Args:
        tensor (torch.Tensor): Tensor to be expanded
        factor (int, optional): Factor to expand the channel dimension by. Defaults to 2.

    Returns:
        torch.Tensor: Expanded tensor

    Usage:
        >>> tensor = torch.rand(1, 3, 224, 224)
        >>> tensor.shape
        torch.Size([1, 3, 224, 224])
        >>> tensor = expand_channels(tensor, factor=2)
        >>> tensor.shape
        torch.Size([1, 6, 224, 224])

    """
    return rearrange(tensor, "b c ... -> b (c factor) ...", factor=factor)


x = torch.rand(1, 3, 224, 224)
model = expand_channels(x, factor=2)
print(model.shape)
