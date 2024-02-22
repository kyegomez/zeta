from typing import Optional

from einops import rearrange
from torch import Tensor, nn

from zeta.nn.modules.ssm import SSM


class VSSBlock(nn.Module):
    """
    VSSBlock is a module that implements a Variational State Space (VSS) block.

    PAPER: https://arxiv.org/pdf/2401.10166.pdf

    Args:
        dim (int): The input dimension.
        d_state (int): The dimension of the state.
        dim_head (int): The dimension of each head in the multi-head attention mechanism.
        heads (int): The number of attention heads.
        dt_rank (int): The rank of the dynamic tensor.
        dim_inner (Optional[int]): The inner dimension of the feed-forward network. Defaults to None.

    Attributes:
        dim (int): The input dimension.
        d_state (int): The dimension of the state.
        dim_head (int): The dimension of each head in the multi-head attention mechanism.
        heads (int): The number of attention heads.
        dt_rank (int): The rank of the dynamic tensor.
        dim_inner (int): The inner dimension of the feed-forward network.
        scale (float): The scaling factor for the attention weights.
        norm (nn.LayerNorm): The layer normalization module.
        depthwise_conv (nn.Conv1d): The depthwise convolution layer.
        proj (nn.Linear): The linear projection layer.
        ssm (SSM): The Variational State Space Model (SSM) module.

    """

    def __init__(
        self,
        dim: int,
        d_state: int,
        dim_head: int,
        heads: int,
        dt_rank: int,
        dim_inner: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.dim_head = dim_head
        self.heads = heads
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner if dim_inner is not None else dim * 4

        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.depthwise_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
        )
        self.proj = nn.Linear(dim, dim)
        self.ssm = SSM(
            in_features=dim,
            dt_rank=dt_rank,
            dim_inner=dim_inner,
            d_state=d_state,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the VSSBlock module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the VSSBlock module.
        """
        skip = x

        x = self.norm(x)

        # Linear projection
        x = self.proj(x)

        linear_skip = x
        linear_skip = self.proj(linear_skip)

        # Depthwise convolution
        x = rearrange(x, "b n (h d) -> b (n h) d", h=self.heads)
        x = self.depthwise_conv(x)
        x = rearrange(x, "b (n h) d -> b n (h d)", h=self.heads)

        # SSM
        x = self.ssm(x)

        # Layernorm
        x = self.norm(x)

        # Matmul with layernorm and skip connection
        x = x @ linear_skip

        # linear
        x = self.proj(x)

        # Addition
        x + skip
