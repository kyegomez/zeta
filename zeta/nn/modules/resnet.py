from torch import nn
from einops.layers.torch import Rearrange, Reduce
import math


def make_layer(inplanes, planes, block, n_blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        # output size won't match input so adjust residual
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
        )
        nn.BatchNorm2d(planes * block.expansion),

    return nn.Sequential(
        block(inplanes, planes, stride, downsample),
        *[block(planes * block.expansion, planes) for _ in range(1, n_blocks)],
    )


class ResNet(nn.Module):
    """
    Resnet implementation.

    Usage:
        from zeta.nn import ResNet

        x = torch.randn(1, 3, 224, 224)
        net = ResNet()
        net(x)



    """

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()

        e = block.expansion

        self.resnet = nn.Sequential(
            Rearrange("b c h w -> b c h w", c=3, h=224, w=224),
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            make_layer(64, 64, block, layers[0], stride=1),
            make_layer(64 * e, 128, block, layers[1], stride=2),
            make_layer(128 * e, 256, block, layers[2], stride=2),
            make_layer(256 * e, 512, block, layers[3], stride=2),
            # combind avg pool and view in one averaging operation
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(512 * e, num_classes),
        )

    def __call__(self):
        """
        Call the forward pass.
        """
        for m in self.resnet.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return self.resent
