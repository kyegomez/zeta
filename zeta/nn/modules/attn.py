import math
import torch


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
) -> torch.Tensor:
    """
    Compute scaled dot product attention.

    Args:
        query (torch.Tensor): The query tensor of shape (..., L, H).
        key (torch.Tensor): The key tensor of shape (..., S, H).
        value (torch.Tensor): The value tensor of shape (..., S, D).
        attn_mask (torch.Tensor, optional): The attention mask tensor of shape (..., L, S).
        dropout_p (float, optional): The dropout probability. Default is 0.0.
        is_causal (bool, optional): Whether to use causal attention. Default is False.
        scale (float, optional): The scale factor for the attention weights. Default is None.

    Returns:
        torch.Tensor: The attention weights tensor of shape (..., L, S) multiplied by the value tensor.

    """
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
