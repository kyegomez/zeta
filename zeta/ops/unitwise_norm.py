import torch


def unitwise_norm(x):
    """
    Unitwise norm

    Args:
        x (torch.Tensor): input tensor


    Example:
    >>> x = torch.randn(10, 10)
    >>> unitwise_norm(x)


    """
    if (len(torch.squeeze(x).shape)) <= 1:
        axis = 0
        keepdims = False
    elif len(x.shape) in [2, 3]:
        axis = 1
        keepdims = True
    elif len(x.shape) == 4:
        axis = [1, 2, 4]
        keepdims = True
    else:
        raise ValueError(
            f"Got a parameter with len(shape) not in [1, 2, 3, 5] {x}"
        )

    return torch.sqrt(torch.sum(torch.square(x), axis=axis, keepdim=keepdims))
