import torch
from torch import nn


def _round_filters(filters, width_mult):
    """
    Scale the number of filters based on the width multiplier.

    Parameters
    ----------
    filters : int
        the original number of filters
    width_mult : float
        the width multiplier

    Returns
    -------
    int
        the scaled number of filters
    """
    return int(filters * width_mult)


class ConvBNReLU(nn.Sequential):
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


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block.

    Parameters
    ---------
    in_planes : int
        the number of input channels
    reduced_dim : int
        the number of channels after the first convolution

    Attributes
    ----------
    se : nn.Sequential
        the sequential layers of the Squeeze-and-Excitation block

    Methods
    -------
    forward(x)

    Example:
    --------
    >>> x = torch.randn(1, 3, 256, 256)
    >>> model = SqueezeExcitation(3, 1)
    >>> output = model(x)
    >>> print(output.shape)



    """

    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass for the Squeeze-and-Excitation block."""
        return x * self.se(x)


class MBConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        expand_ratio,
        stride,
        kernel_size,
        reduction_ratio=4,
    ):
        super(MBConv, self).__init__()
        self.stride = stride
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        self.conv = nn.Sequential(
            (
                # pw
                ConvBNReLU(in_planes, hidden_dim, 1)
                if expand_ratio != 1
                else nn.Identity()
            ),
            # dw
            ConvBNReLU(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride=stride,
                groups=hidden_dim,
            ),
            # se
            SqueezeExcitation(hidden_dim, reduced_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    """
    EfficientNet model.

    Parameters
    ----------
    width_mult : float
        the width multiplier

    Attributes
    ----------
    features : nn.Sequential
        the sequential layers of the model
    avgpool : nn.AdaptiveAvgPool2d
        the adaptive average pooling layer
    classifier : nn.Linear
        the linear layer

    Methods
    -------
    forward(x)

    Example:
    >>> x = torch.randn(1, 3, 256, 256)
    >>> model = EfficientNet()
    >>> output = model(x)
    >>> print(output.shape)

    """

    def __init__(self, width_mult=1.0):
        super(EfficientNet, self).__init__()
        # scale dimensions
        input_channel = _round_filters(32, width_mult)
        last_channel = _round_filters(1280, width_mult)

        # define network structure
        self.features = nn.Sequential(
            ConvBNReLU(3, input_channel, 3, stride=2),
            MBConv(input_channel, 16, 1, 1, 3),
            MBConv(16, 24, 6, 2, 3),
            MBConv(24, 40, 6, 2, 5),
            MBConv(40, 80, 6, 2, 3),
            MBConv(80, 112, 6, 1, 5),
            MBConv(112, 192, 6, 2, 5),
            MBConv(192, 320, 6, 1, 3),
            ConvBNReLU(320, last_channel, 1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, 1000)

    def forward(self, x):
        """
        Computes the forward pass for the EfficientNet model.

        Parameters
        ----------
        x : torch.Tensor
            a 4D or 5D tensor containing the input data

        Returns
        -------
        torch.Tensor
            a 4D or 5D tensor containing the computed features
        """
        if len(x.shape) == 5:
            # If the input is a 5D tensor, reshape it to 4D by combining the batch and frames dimensions
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if len(x.shape) == 2 and "b" in locals() and "t" in locals():
            x = x.view(b, t, -1)
        return x


# x = torch.randn(1, 3, 256, 256)
# model = EfficientNet()
# output = model(x)
# print(output.shape)
