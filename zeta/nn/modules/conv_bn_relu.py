
from torch import nn

class ConvBNReLU(nn.Sequential):
    """
    A conv layer followed by batch normalization and ReLU activation.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int, optional): Stride of the convolution. Default is 1.
        groups (int, optional): Number of groups for conv. Default is 1.
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )
    
    def forward(self, x):
        # Placeholder code to access the 'x' variable
        return x
        