import torch
from torch import nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class ShuffleNet(nn.Module):
    """

    ShuffleNet implementation.


    Usage:
        from zeta.nn import ShuffleNet

        x = torch.randn(1, 3, 224, 224)
        net = ShuffleNet()
        net(x)


    """

    def __init__(
        self,
        in_channels,
        out_channels,
        groups=3,
        grouped_conv=True,
        combine="add",
    ):
        super().__init__()
        first_1x1_groups = groups if grouped_conv else 1
        bottle_neck_channels = out_channels // 4

        self.combine = combine

        if combine == "add":
            # shuffleunit
            self.left = Rearrange("...->...")  # identity
            depthwise_stride = 1

        else:
            # shuffleunit figure 2x
            self.left = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            depthwise_stride = 2

            # Ensure output of concat has the same channels as original output channels
            out_channels -= in_channels
            assert out_channels > 0

        conv1x1 = nn.Conv2d(
            in_channels,
            bottle_neck_channels,
            kernel_size=1,
            groups=first_1x1_groups,
            bias=False,
        )

        conv3x3_depthwise = nn.Conv2d(
            bottle_neck_channels,
            bottle_neck_channels,
            kernel_size=3,
            stride=depthwise_stride,
            padding=1,
            groups=bottle_neck_channels,
            bias=False,
        )

        self.right = nn.Sequential(
            # use a 1x1 grouped or non grouped convolution to reduce input channels
            # to bottleneck channels as in a resnet bottleneck module
            conv1x1(in_channels, bottle_neck_channels, groups=first_1x1_groups),
            nn.BatchNorm2d(bottle_neck_channels),
            nn.ReLU(inplace=True),
            # channels shuffle
            Rearrange("b (c1 c2) h w -> b (c2 c1) h w", c1=groups),
            # 3x3 depthwise convolution
            conv3x3_depthwise(
                bottle_neck_channels,
                stride=depthwise_stride,
                groups=bottle_neck_channels,
            ),
            nn.BatchNorm2d(bottle_neck_channels),
            # use 1x1 grouped conbolution to expand from
            # bottleneck channels to out channels
            conv1x1(bottle_neck_channels, out_channels, groups=groups),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        """
        Run the forward pass.

        Usage:
            net = ShuffleNet()
            net(x)
        """
        if self.combine == "add":
            combined = self.left(x) + self.right(x)
        else:
            combined = torch.cat([self.left(x), self.right(x)], dim=1)
        return F.relu(combined, inplace=True)
