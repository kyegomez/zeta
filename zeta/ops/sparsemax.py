import torch
from torch import Tensor


def sparsemax(x: Tensor):
    """
    A PyTorch implementation of the sparsemax function.

    Args:
        x (torch.Tensor): The x tensor.

    Returns:
        torch.Tensor: The output of the sparsemax function.
        
    Example:
    >>> x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    >>> sparsemax(x)
    tensor([0., 0., 0., 1., 1.])
    """
    dim = x.dim() - 1
    number_of_logits = x.size(dim)

    x = x - torch.max(x, dim=dim, keepdim=True)[0].expand_as(x)
    zs = torch.sort(x=x, dim=dim, descending=True)[0]
    range = torch.arange(start=1, end=number_of_logits + 1, device=x.device).view(1, -1)
    range = range.expand_as(zs)

    bound = 1 + range * zs
    cumulative_sum_zs = torch.cumsum(zs, dim)
    is_gt = torch.gt(bound, cumulative_sum_zs).type(x.type())
    k = torch.max(is_gt * range, dim, keepdim=True)[0]

    zs_sparse = is_gt * zs
    taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
    taus = taus.expand_as(x)
    output = torch.max(torch.zeros_like(x), x - taus)
    return output