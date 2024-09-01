import torch
from torch import nn


# Example usage of the IterativeCrossSelfAttention class
class PreNorm(nn.Module):
    """Prenorm

    Args:
        dim (_type_): _description_
        fn (_type_): _description_

    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, context=None):
        """Forward pass of prenorm

        Args:
            x (_type_): _description_
        """
        return self.fn(self.norm(x), context=context)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        qk_norm: bool = True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        )

        self._qk_norm = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        if context is None:
            context = x

        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim=-1)
        k, v = kv[0], kv[1]

        if self.qk_norm:
            q, k = self._qk_norm(q), self._qk_norm(k)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = self.to_out(out)
        return out


class IterativeCrossSelfAttention(nn.Module):
    """Iterative

    Args:
        dim (_type_): _description_
        depth (_type_): _description_
        heads (_type_): _description_
        dim_head (_type_): _description_
        dropout (float, optional): _description_. Defaults to 0.1.

    Methods:
        forward(x, context=None): _description_

    Examples:
    """

    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        dropout=0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PreNorm(
                    dim,
                    CrossAttention(
                        dim, heads=heads, dim_head=dim_head, dropout=dropout
                    ),
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        """Forward pass of IterativeCrossSelfAttention

        Args:
            x (torch.Tensor): _description_
            context (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        for layer in self.layers:
            x = layer(x, context=context) + x
        return x


# import torch

# # Example usage of the IterativeCrossSelfAttention class
# if __name__ == "__main__":
#     batch_size = 8
#     seq_len = 16  # Sequence length of the input embeddings
#     latent_seq_len = 16  # Sequence length of the latent array (could be different from input sequence length)
#     dim = 512  # Dimensionality of the input embeddings and latent array
#     heads = 8  # Number of attention heads
#     dim_head = 64  # Dimensionality of each attention head
#     depth = 6  # Number of cross-attention layers

#     # Initialize the IterativeCrossSelfAttention module
#     iter_cs_attn = IterativeCrossSelfAttention(dim, depth, heads, dim_head)

#     # Create random tensors for the input embeddings and the latent array
#     input_embeddings = torch.rand(batch_size, seq_len, dim)
#     latent_array = torch.rand(batch_size, latent_seq_len, dim)

#     # Pass the input embeddings and the latent array through the IterativeCrossSelfAttention module
#     output_embeddings = iter_cs_attn(input_embeddings, latent_array)

#     print("Output embeddings shape:", output_embeddings.shape)
