import torch
import torch.nn.functional as F
from torch import nn

class Exo(nn.Module):
    """

    Exo activation function
    -----------------------

    Exo is a new activation function that is a combination of linear and non-linear parts.

    Formula
    -------
    .. math::
        f(x) = \\sigma(x) \\cdot x + (1 - \\sigma(x)) \\cdot tanh(x)

    Parameters
    ----------
    alpha : float
        Alpha value for the activation function. Default: 1.0

    Examples
    --------
    >>> m = Exo()
    >>> input = torch.randn(2)
    >>> output = m(input)



    """
    def __init__(self, alpha=1.0):
        """INIT function."""
        super(Exo, self).__init__()
    
    def forward(self, x):
        """Forward function."""
        gate = torch.sigmoid(x)
        linear_part = x
        non_linear_part = torch.tanh(x)
        return gate * linear_part + (1 - gate) * non_linear_part
    

