import torch
from torch import nn


class MMFusionFFN(nn.Module):
    r"""Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        return self.net(input)
