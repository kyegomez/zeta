# QKV Normalization

from torch import nn


def qkv_norm(
    q,
    k,
    v,
):
    """Apply QKV normalization.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.

    Returns:
        torch.Tensor: Normalized query, key, and value tensors.
    """
    q = nn.LayerNorm(q.size())(q)
    k = nn.LayerNorm(k.size())(k)
    v = nn.LayerNorm(v.size())(v)
    return q, k, v


def qk_norm(
    q,
    k,
):
    """Apply QK normalization.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.

    Returns:
        torch.Tensor: Normalized query, key, and value tensors.
    """
    q = nn.LayerNorm(q.size())(q)
    k = nn.LayerNorm(k.size())(k)
    return q, k