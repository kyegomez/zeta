import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapedAttention(nn.Module):
    """
    ShapedAttention module as described in the provided text.
    This module implements a Transformer attention mechanism with
    simplified attention sub-block (SAS) and shaped attention.

    Parameters:
    - dim: The dimensionality of the input feature space.
    - heads: The number of attention heads.
    - dropout: The dropout rate to be applied to the attention scores.
    """

    def __init__(self, dim, heads, dropout=0.1):
        super(ShapedAttention, self).__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        # Define the key, query, and value matrices for the attention
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        # Shaped attention specific parameters
        self.alpha = nn.Parameter(torch.ones(1, heads, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, heads, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, heads, 1, 1))

        # Centering matrix (not trained)
        self.register_buffer("C", torch.zeros(heads, 1, 1))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Split the input into multiple heads
        B, T, _ = x.shape
        q = self.query(x).view(B, T, self.heads, -1).transpose(1, 2)
        k = self.key(x).view(B, T, self.heads, -1).transpose(1, 2)
        v = self.value(x).view(B, T, self.heads, -1).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply shaped attention modifications
        attn = (
            self.alpha * torch.eye(T).to(attn.device)
            + self.beta * attn
            - self.gamma * self.C
        )

        # Apply attention to values and combine heads
        x = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)

        return self.dropout(x)


# # Example usage
# dim = 768
# heads = 8
# dropout = 0.1

# shaped_attention = ShapedAttention(dim, heads, dropout)

# x = torch.randn(1, 32, 768)

# out = shaped_attention(x)
# print(out)
