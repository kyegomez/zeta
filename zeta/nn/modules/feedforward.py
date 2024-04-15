from torch import nn
import torch.nn.functional as F
from zeta.nn.modules.glu import GLU
from zeta.nn.modules.swiglu import SwiGLU
from typing import Optional


class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


def exists(val):
    return val is not None


def default(val, default_val):
    return default_val if val is None else val


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: Optional[int] = None,
        dim_out: Optional[int] = None,
        mult: Optional[int] = 4,
        glu: Optional[bool] = False,
        glu_mult_bias: Optional[bool] = False,
        swish: Optional[bool] = False,
        relu_squared: Optional[bool] = False,
        post_act_ln: Optional[bool] = False,
        dropout: Optional[float] = 0.0,
        no_bias: Optional[bool] = False,
        zero_init_output: Optional[bool] = False,
        custom_act: Optional[nn.Module] = None,
        swiglu: Optional[bool] = False,
    ):
        """
        FeedForward module that applies a series of linear transformations and activations.

        Args:
            dim (int): Input dimension.
            dim_out (int, optional): Output dimension. Defaults to None.
            mult (int, optional): Multiplier for the inner dimension. Defaults to 4.
            glu (bool, optional): Whether to use Gated Linear Units (GLU). Defaults to False.
            glu_mult_bias (bool, optional): Whether to use bias in the GLU operation. Defaults to False.
            swish (bool, optional): Whether to use Swish activation. Defaults to False.
            relu_squared (bool, optional): Whether to use squared ReLU activation. Defaults to False.
            post_act_ln (bool, optional): Whether to apply Layer Normalization after the activation. Defaults to False.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            no_bias (bool, optional): Whether to use bias in the linear transformations. Defaults to False.
            zero_init_output (bool, optional): Whether to initialize the last linear layer to 0. Defaults to False.
            custom_act (nn.Module, optional): Custom activation module. Defaults to None.
            swiglu (bool, optional): Whether to use SwiGLU activation. Defaults to False.
        """
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        elif custom_act is not None:
            activation = custom_act
        elif swiglu:
            activation = SwiGLU()
        else:
            activation = nn.GELU()

        if glu:
            project_in = GLU(
                dim, inner_dim, activation, mult_bias=glu_mult_bias
            )
        else:
            project_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias=not no_bias), activation
            )

        if post_act_ln:
            self.ff = nn.Sequential(
                project_in,
                nn.LayerNorm(inner_dim),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out, bias=not no_bias),
            )
        else:
            self.ff = nn.Sequential(
                project_in,
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out, bias=not no_bias),
            )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        """
        Forward pass of the feedforward network

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.ff(x)
