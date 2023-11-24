from collections import OrderedDict
import torch
from torch import nn


class ClipBottleneck(nn.Module):
    """
    ClipBottleneck is a bottleneck block with a stride of 1 and an avgpool layer after the second conv layer.

    Args:
        inplanes (int): Number of input channels
        planes (int): Number of output channels
        stride (int): Stride of the first conv layer. Default: 1


    Attributes:
        expansion (int): Expansion factor of the block. Default: 4

    Usage:
    >>> block = ClipBottleneck(64, 256, stride=2)
    >>> x = torch.rand(1, 64, 32, 32)
    >>> out = block(x)
    >>> out.shape


    """

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
    ):
        super().__init__()

        # All conv layers have stride 1 an agvpool is performaned after the second conv layer
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * ClipBottleneck.expansion:
            # downsampling layer is prepended with an avgpool layer
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes, planes * self.expansion, 1, bias=False
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        """Forward pass of the block"""
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out
