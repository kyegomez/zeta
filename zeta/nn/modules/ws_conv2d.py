import torch
from torch import nn, Tensor
import torch.nn.functional as F


class WSConv2d(nn.Conv2d):
    """
    Weight Standardized Convolutional 2D Layer.

    This class inherits from `nn.Conv2d` and adds weight standardization to the convolutional layer.
    It normalizes the weights of the convolutional layer to have zero mean and unit variance along
    the channel dimension. This helps in stabilizing the training process and improving generalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (float, optional): Stride of the convolution. Default is 1.
        padding (int or tuple, optional): Padding added to the input. Default is 0.
        dilation (int, optional): Spacing between kernel elements. Default is 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default is 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Default is True.
        padding_mode (str, optional): Type of padding. Default is "zeros".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: float = 1,
        padding=0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super(WSConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

        nn.init.xavier_normal_(self.weight)

        # Params
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer(
            "eps", torch.tensor(1e-4, requires_grad=False), persistent=False
        )
        self.register_buffer(
            "fan_in",
            torch.tensor(
                self.weight.shape[1:].numel(), requires_grad=False
            ).type_as(self.weight),
            persistent=False,
        )

    def standardized_weights(self):
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
        scale = torch.rsqrt(torch.maximum(var * self.fan_in, self.eps))
        return (self.weight - mean) * scale * self.gain

    def forward(self, x: Tensor):
        return F.conv2d(
            input=x,
            weight=self.standardized_weights(),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
