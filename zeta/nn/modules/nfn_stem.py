import torch
from torch import nn, Tensor
from zeta.nn.modules.ws_conv2d import WSConv2d
from typing import List

class NFNStem(nn.Module):
    """
    NFNStem module represents the stem of the NFN (Neural Filter Network) architecture.
    
    Args:
        in_channels (List[int]): List of input channel sizes for each layer. Default is [3, 16, 32, 64].
        out_channels (List[int]): List of output channel sizes for each layer. Default is [16, 32, 64, 128].
        kernel_size (int): Size of the convolutional kernel. Default is 3.
        stride (List[int]): List of stride values for each layer. Default is [2, 1, 1, 2].
        activation (nn.Module): Activation function to be applied after each convolutional layer. Default is nn.GELU().
        
    Examples:
        >>> x = torch.randn(1, 3, 224, 224)
        >>> model = NFNStem()
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([1, 128, 28, 28])
    """
    def __init__(
        self,
        in_channels: List[int] = [3, 16, 32, 64],
        out_channels: List[int] = [16, 32, 64, 128],
        kernel_size: int = 3,
        stride: List[int] = [2, 1, 1, 2],
        activation: nn.Module = nn.GELU(),
    ):
        super(NFNStem, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv0 = WSConv2d(
            in_channels=self.in_channels[0],
            out_channels=self.out_channels[0],
            kernel_size=3,
            stride = self.stride[0],
        )
        self.conv1 = WSConv2d(
            in_channels=self.in_channels[1],
            out_channels=self.out_channels[1],
            kernel_size=kernel_size,
            stride=self.stride[1]
        )
        self.conv2 = WSConv2d(
            in_channels=self.in_channels[2],
            out_channels=self.out_channels[2],
            kernel_size=kernel_size,
            stride=self.stride[2]
        )
        self.conv3 = WSConv2d(
            in_channels=self.in_channels[3],
            out_channels=out_channels[3],
            kernel_size=kernel_size,
            stride=self.stride[3]
        )
        
    def forward(self, x: Tensor):
        """Forward pass of the NFNStem module.

        Args:
            x (Tensor): _description_

        Returns:
            _type_: _description_
        """
        out = self.activation(self.conv0(x))
        out = self.activation(self.conv1(out))
        out = self.activation(self.conv2(out))
        out = self.conv3(out)
        return out

x = torch.randn(1, 3, 224, 224)
model = NFNStem()
out = model(x)
print(out)