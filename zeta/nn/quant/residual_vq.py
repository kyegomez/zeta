import torch
from torch import nn


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer.

    Args:
        dim (int): _description_
        dim_out (int): _description_
        n_embed (int): _description

    Example:
        >>> x = torch.randn(2, 4)
        >>> model = ResidualVectorQuantizer(4, 4, 4)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([2, 4])
    """

    def __init__(self, dim, dim_out, n_embed):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.n_embed = n_embed
        self.embed = nn.Embedding(n_embed, dim)
        self.proj = nn.Linear(dim, dim_out)

    def forward(self, x):
        """Forward pass of the ResidualVectorQuantizer module.

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Compute distances to embedding vectors
        dists = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1)
        )

        # Find the closest embedding for each input vector
        _, embed_ind = dists.min(1)
        embed_onehot = torch.zeros_like(dists).scatter_(
            1, embed_ind.view(-1, 1), 1
        )
        embed_ind = embed_onehot @ self.embed.weight

        # Compute residual
        residual = self.proj(x - embed_ind)

        # Add residual to the input
        x = x + residual

        return x


# x = torch.randn(2, 4)
# model = ResidualVectorQuantizer(4, 4, 4)
# out = model(x)
# print(out.shape)
