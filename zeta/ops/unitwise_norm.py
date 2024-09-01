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
    if len(torch.squeeze(x).shape) <= 1:
        pass
    elif len(x.shape) in [2, 3]:
        pass
    elif len(x.shape) == 4:
        pass
    else:
        raise ValueError(
            f"Got a parameter with len(shape) not in [1, 2, 3, 5] {x}"
        )
