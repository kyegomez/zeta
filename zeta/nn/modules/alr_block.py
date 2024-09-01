import torch
import torch.nn as nn


class FeedForward(nn.Module):
    # Assuming FeedForward class is something like this
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim, in_dim
            ),  # Ensuring the output dimension is the same as input
        )

    def forward(self, x):
        return self.net(x)


class ALRBlock(nn.Module):
    """
    ALRBlock class
    A transformer like layer that uses feedforward networks instead of self-attention

    Args:
        dim (int): Input dimension
        hidden_dim (int): Hidden dimension
        dropout (float): Dropout rate

    Usage:
    >>> model = ALRBlock(512, 2048, 0.1)
    >>> x = torch.randn(1, 1024, 512)
    >>> model(x).shape

    """

    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.ffn = FeedForward(
            dim * 3, hidden_dim, dropout
        )  # Adjusted for 3 * dim
        self.ff = FeedForward(dim, hidden_dim, dropout)

        self.to_q_proj = nn.Linear(dim, dim)
        self.to_k_proj = nn.Linear(dim, dim)
        self.to_v_proj = nn.Linear(dim, dim)

        self.norm_ffn = nn.LayerNorm(dim)  # Adjusted for 3 * dim
        self.norm_ff = nn.LayerNorm(dim)

        self.proj_out = nn.Linear(dim * 3, dim)

    def forward(self, x):
        """Forward method of ALRBlock"""
        q, k, v = self.to_q_proj(x), self.to_k_proj(x), self.to_v_proj(x)

        qkv = torch.cat((q, k, v), dim=-1)

        ffn = self.ffn(qkv)
        ffn_projected = self.proj_out(ffn)
        norm_ffn = self.norm_ffn(ffn_projected) + x

        ff = self.ff(norm_ffn)
        ff_norm = self.norm_ff(ff)

        out = ff_norm + x

        return out
