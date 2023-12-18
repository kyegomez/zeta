import torch
import torch.nn as nn


# Basic Block for ResNet
class BasicBlock(nn.Module):
    """BasicBlock


    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride of the convolutional layer
        kernel_size (int): Kernel size of the convolutional layer
        padding (int): Padding of the convolutional layer
        bias (bool): Bias of the convolutional layer

    Examples:
    >>> from zeta.nn.modules.res_net import BasicBlock
    >>> import torch
    >>> x = torch.randn(5, 10)
    >>> swiglu = BasicBlock(10, 20)
    >>> swiglu(x).shape
    torch.Size([5, 10])

    """

    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
        *args,
        **kwargs,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=bias,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x: torch.Tensor):
        """Forward

        Args:
            x torch.Tensor: Input tensor

        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# Full ResNet
class ResNet(nn.Module):
    """ResNet

    Args:
        block (_type_): _description_
        num_blocks (_type_): _description_
        num_classes (int): Number of classes
        kernel_size (int): Kernel size of the convolutional layer
        stride (int): Stride of the convolutional layer
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Examples:
    >>> from zeta.nn.modules.res_net import ResNet
    >>> import torch
    >>> x = torch.randn(5, 10)
    >>> swiglu = ResNet(10, 20)
    >>> swiglu(x).shape
    torch.Size([5, 10])


    """

    def __init__(
        self,
        block,
        num_blocks,
        num_classes: int = 1000,
        kernel_size: int = 3,
        stride: int = 2,
        *args,
        **kwargs,
    ):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=stride, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=1
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=stride)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=stride)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=stride)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=stride)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Make layer

        Args:
            block (_type_): _description_
            out_channels (_type_): _description_
            num_blocks (_type_): _description_
            stride (_type_): _description_

        Returns:
            _type_: _description_
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Forward

        Args:
            x torch.Tensor: Input tensor
        """
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# model = ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10)

# x = torch.randn(1, 3, 224, 224)

# print(model(x).shape)
