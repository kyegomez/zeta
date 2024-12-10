import torch
from torch import nn, Tensor
from typing import Callable


class GLU(nn.Module):
    """
    GLU (Gated Linear Unit) module.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        activation (Callable[[Tensor], Tensor]): Activation function to be applied to the gate.
        mult_bias (bool, optional): Whether to multiply the bias term. Defaults to False.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: Callable[[Tensor], Tensor],
        mult_bias: bool = False,
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.0

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate) * self.mult_bias
