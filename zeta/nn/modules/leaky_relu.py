import torch
from torch import nn


class LeakyRELU(nn.Module):
    """LeakyReLU activation function.

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """

    __constants__ = ["inplace", "negative_slope"]
    inplace: bool
    negative_sloop: float

    def __init__(
        self,
        negative_slope: float = 1e-2,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the LeakyReLU module.

        Args:
            input (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return torch.where(input >= 0.0, input, input * self.negative_slope)

    def extra_repr(self) -> str:
        """Extra information about this module.

        Returns:
            str: _description_
        """
        inplace_str = ", inplace=True" if self.inplace else ""
        return f"negative_slope={self.negative_slope}{inplace_str}"
