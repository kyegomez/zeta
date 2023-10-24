from einops.layers.torch import Rearrange
from torch import nn


class PatchEmbeddings(nn.Module):
    """
    Patch embeddings for images.

    Parameters:
    - dim_in (int): The input dimension.
    - dim_out (int): The output dimension.
    - seq_len (int): The sequence length.

    Attributes:
    - embedding (nn.Module): The embedding layer.

    Example:

        >>> module = PatchEmbeddings(3, 4, 5)
        >>> x = torch.randn(2, 3, 5, 5)
        >>> y = module(x)
        >>> y.shape
        torch.Size([2, 20, 5])

    """

    def __init__(self, dim_in, dim_out, seq_len):
        super().__init__()
        self.embedding = nn.Sequential(
            Rearrange("... rd -> ... (r d)"),
            nn.LayerNorm(seq_len * dim_in),
            nn.Linear(seq_len * dim_in, dim_out),
            nn.LayerNorm(dim_out),
        )

    def forward(self, x):
        return self.embedding(x)
