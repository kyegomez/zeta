import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def forward(self, x):
        """Forward

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class SwiGLUStacked(nn.Module):
    """SwiGLUStacked

    Args:
        nn (_type_): _description_

    Examples:
    >>> from zeta.nn.modules.swiglu import SwiGLUStacked
    >>> import torch
    >>> x = torch.randn(5, 10)
    >>> swiglu = SwiGLUStacked(10, 20)
    >>> swiglu(x).shape
    torch.Size([5, 10])
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        dropout: float = None,
        bias: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x):
        """Forward

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return x
