import torch
from einops import rearrange
from torch import nn


class MultiModalCrossAttention(nn.Module):
    """
    Enhanced CrossAttention module with conditional layer normalization, lambda masking, and dropout.


    Args:
        dim: Dimension of the model.
        heads: Number of attention heads.
        context_dim: Dimension of the context.
        dim_head: Dimension of each attention head.
        dropout: Dropout rate.
        qk: Whether to use conditional layer normalization.
        post_attn_norm: Whether to use post-attention

    Examples:
        import torch
        import torch.nn as nn
        from zeta.nn.attention.cross_attn_images import CrossAttention
        x = torch.randn(1, 32, 1024)
        context = torch.randn(1, 32, 1024)
        attn = CrossAttention(1024, 8, 1024)
        out = attn(x, context)
        out.shape
        torch.Size([1, 32, 1024])
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        context_dim: int,
        dim_head=64,
        dropout=0.1,
        qk: bool = False,
        post_attn_norm: bool = False,
        attention_strategy: str = None,  # "average",
        mask=None,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        self.qk = qk
        self.post_attn_norm = post_attn_norm
        self.attention_strategy = attention_strategy
        self.mask = mask
        self.context_dim = context_dim

        # Linear layers for q, k, v
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(dim_head * heads, dim), nn.Dropout(dropout)
        )

    def forward(self, x, context):
        # Compute query, key, value
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Optional conditional layer normalization
        if self.qk:
            q = self.norm_q(q)
            k = self.norm_k(k)

        # Reshape for multi-head attention
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            (q, k, v),
        )

        # Scaled dot-product attention
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        # Optional masking
        if self.mask is not None:
            dots.masked_fill_(~self.mask, float("-inf"))

        # Softmax and dropout on attention weights
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # Compute output
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        # Average or concatenate heads based on strategy
        if self.attention_strategy == "average":
            out = out.mean(dim=1)

        # Post-attention normalization
        if self.post_attn_norm:
            out = self.norm_post_attn(out)

        # Output projection
        return self.to_out(out)
