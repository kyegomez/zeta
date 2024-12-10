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
