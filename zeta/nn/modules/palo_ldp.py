
from torch import Tensor, nn
from zeta.utils.log_pytorch_op import log_torch_op

class PaloLDP(nn.Module):
    """
    Implementation of the PaloLDP module.

    Args:
        dim (int): The dimension of the input tensor.
        channels (int, optional): The number of input channels. Defaults to 1.
    """

    def __init__(
        self,
        dim: int,
        channels: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.channels = channels

        self.pointwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.gelu = nn.GELU()

        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
        )

        # LayerNorm
        self.norm = nn.LayerNorm(dim)

        # Depthwise convolution with stride = 2
        self.depthwise_conv_stride = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=channels,
        )

    @log_torch_op()
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the PaloLDP module.

        Args:
            x (Tensor): The input tensor of shape (B, C, H, W).

        Returns:
            Tensor: The output tensor of shape (B, C, H', W').
        """
        b, c, h, w = x.shape

        x = self.pointwise_conv(x)
        print(x.shape)  # torch.Size([2, 1, 4, 4]

        x = self.gelu(x)
        print(x.shape)  # torch.Size([2, 1, 4, 4]

        x = self.pointwise_conv(x)
        print(x.shape)  # torch.Size([2, 1, 4, 4]


        # Depthwise convolution with 1 stide
        x = self.depthwise_conv(x)
        print(x.shape)

        # Norm
        x = self.norm(x)
        print(x.shape)

        # Pointwise convolution
        x = self.pointwise_conv(x)
        print(x.shape)

        # Norm
        x = self.norm(x) #+ skip
        print(x.shape)

        # Depthwise convolution with 2 stide
        x = self.depthwise_conv_stride(x)
        print(x.shape)

        # Norm
        b, c, h, w = x.shape
        # x = self.norm(x)
        x = nn.LayerNorm(w)(x)

        # Pointwise convolution
        x = self.pointwise_conv(x)

        # Norm
        b, c, h, w = x.shape
        x = nn.LayerNorm(w)(x)

        return x


