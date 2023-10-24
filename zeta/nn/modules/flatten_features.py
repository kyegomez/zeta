import torch
from einops import rearrange


def flatten_features(feature_map):
    """
    Flaten the spatial dimensions of a feature map

    Einstein summation notation:
        'b' = batch size
        'h' = image height
        'w' = image width
        'c' = number of channels

    Args:
        feature_map (torch.Tensor): Feature map to be flattened

    Returns:
        torch.Tensor: Flattened feature map

    Example:
        >>> feature_map = torch.rand(1, 3, 224, 224)
        >>> feature_map.shape
        torch.Size([1, 3, 224, 224])
        >>> feature_map = flatten_features(feature_map)
        >>> feature_map.shape
        torch.Size([1, 150528])
    """
    return rearrange(feature_map, "b c h w -> b (c h w)")


# #random
# x = torch.rand(1, 3, 224, 224)
# model = flatten_features(x)
# print(model.shape)
