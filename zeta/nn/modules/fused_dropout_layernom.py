import torch
from torch import nn


class FusedDropoutLayerNorm(nn.Module):
    """FusedDropoutLayerNorm

    Args:
        dim (int): Input dimension
        dropout (float, optional): Dropout. Defaults to 0.1.
        eps (float, optional): Epsilon. Defaults to 1e-5.
        elementwise_affine (bool, optional): Elementwise affine. Defaults to True.

    Examples:
        >>> x = torch.randn(1, 512)
        >>> model = FusedDropoutLayerNorm(512)
        >>> out = model(x)
        >>> out.shape
        torch.Size([1, 512])
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Dropout initialization
        self.dropout = nn.Dropout(dropout)

        # LayerNorm initialization
        self.layer_norm = nn.LayerNorm(
            dim, eps=eps, elementwise_affine=elementwise_affine, *args, **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): tensor

        Returns:

        """
        x = self.dropout(x)
        return self.layer_norm(x)
