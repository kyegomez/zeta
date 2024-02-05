from torch import nn


class NormalizationFractral(nn.Module):
    """
    A module that performs normalization using fractal layers.

    Args:
        dim (int): The input dimension.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-8.
        fi (int, optional): The number of fractal layers. Default is 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        fi (int): The number of fractal layers.
        norm (nn.LayerNorm): The initial normalization layer.
        norm_i (nn.LayerNorm): Fractal normalization layers.

    """

    def __init__(
        self, dim: int, eps=1e-8, fi: int = 4, *args, **kwargs  # Fractal index
    ):
        super(NormalizationFractral, self).__init__(*args, **kwargs)
        self.eps = eps
        self.fi = fi

        self.norm = nn.LayerNorm(dim)

        for i in range(fi):
            setattr(self, f"norm_{i}", nn.LayerNorm(dim))

    def forward(self, x):
        """
        Forward pass of the NormalizationFractral module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized output tensor.

        """
        x = self.norm(x)

        for i in range(self.fi):
            norm = getattr(self, f"norm_{i}")
            x = norm(x)

        return x
