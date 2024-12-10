import math

from einops import rearrange
from torch import Tensor, nn

from zeta.nn.modules.simple_mamba import MambaBlock


class UMambaBlock(nn.Module):
    """
    UMambaBlock is a 5d Mamba block that can be used as a building block for a 5d visual model
    From the paper: https://arxiv.org/pdf/2401.04722.pdf

    Args:
        dim (int): The input dimension.
        dim_inner (Optional[int]): The inner dimension. If not provided, it is set to dim * expand.
        depth (int): The depth of the Mamba block.
        d_state (int): The state dimension. Default is 16.
        expand (int): The expansion factor. Default is 2.
        dt_rank (Union[int, str]): The rank of the temporal difference (Δ) tensor. Default is "auto".
        d_conv (int): The dimension of the convolutional kernel. Default is 4.
        conv_bias (bool): Whether to include bias in the convolutional layer. Default is True.
        bias (bool): Whether to include bias in the linear layers. Default is False.

    Examples::
        import torch
        # img:         B, C, H, W, D
        img_tensor = torch.randn(1, 64, 10, 10, 10)

        # Initialize Mamba block
        block = UMambaBlock(dim=64, depth=1)

        # Forward pass
        y = block(img_tensor)
        print(y.shape)

    """

    def __init__(
        self,
        dim: int = None,
        depth: int = 5,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.bias = bias

        # If dt_rank is not provided, set it to ceil(dim / d_state)
        dt_rank = math.ceil(self.dim / 16)
        self.dt_rank = dt_rank

        # If dim_inner is not provided, set it to dim * expand
        dim_inner = dim * expand
        self.dim_inner = dim_inner

        # If dim_inner is not provided, set it to dim * expand
        self.in_proj = nn.Linear(dim, dim_inner, bias=False)
        self.out_proj = nn.Linear(dim_inner, dim, bias=False)

        # Implement 2d convolutional layer
        # 3D depthwise convolution
        self.conv1 = nn.Conv3d(
            in_channels=dim,
            out_channels=dim_inner,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.conv2 = nn.Conv3d(
            in_channels=dim_inner,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        # Init instance normalization
        self.instance_norm = nn.InstanceNorm3d(dim)
        self.instance_norm2 = nn.InstanceNorm3d(dim_inner)

        # Leaky RELU
        self.leaky_relu = nn.LeakyReLU()

        # Layernorm
        self.norm = nn.LayerNorm(dim)

        # Mamba block
        self.mamba = MambaBlock(
            dim=dim,
            depth=depth,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            conv_bias=conv_bias,
            bias=bias,
        )

    def forward(self, x: Tensor):
        """
        B, C, H, W, D
        """
        b, c, h, w, d = x.shape
        input = x
        print(f"Input shape: {x.shape}")

        # Apply convolution
        x = self.conv1(x)
        print(f"Conv1 shape: {x.shape}")

        # # Instance Normalization
        x = self.instance_norm(x) + self.leaky_relu(x)
        print(f"Instance Norm shape: {x.shape}")

        # TODO: Add another residual connection here

        x = self.conv2(x)

        x = self.instance_norm(x) + self.leaky_relu(x)

        x = x + input

        # # Flatten to B, L, C
        x = rearrange(x, "b c h w d -> b (h w d) c")
        print(f"Faltten shape: {x.shape}")
        x = self.norm(x)

        # Maybe use a mamba block here then reshape back to B, C, H, W, D
        x = self.mamba(x)

        # Reshape back to B, C, H, W, D
        x = rearrange(x, "b (h w d) c -> b c h w d", h=h, w=w, d=d)

        return x
