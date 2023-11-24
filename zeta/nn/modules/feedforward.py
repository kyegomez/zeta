from torch import nn

from zeta.structs.attn_layers import GLU
from zeta.structs.transformer import ReluSquared


def exists(val):
    return val is not None


def default(val, default_val):
    return default_val if val is None else val


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)


class FeedForward(nn.Module):
    """
    Feedforward neural network with LayerNorms and GELU activations

    Args:
        dim (int): Input dimension
        hidden_dim (int): Hidden dimension
        dropout (float): Dropout probability

    Usage:
    >>> model = FeedForward(768, 2048, 0.1)
    >>> x = torch.randn(1, 768)
    >>> model(x).shape

    """

    def __init__(
        self,
        dim: int,
        dim_out: int = None,
        mult=4,
        glu=False,
        glu_mult_bias=False,
        swish=False,
        relu_squared=False,
        post_act_ln=False,
        dropout: float = 0.0,
        no_bias=False,
        zero_init_output=False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
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
