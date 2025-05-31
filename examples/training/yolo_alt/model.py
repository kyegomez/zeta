import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GhostModule(nn.Module):
    def __init__(
        self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True
    ):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp,
                init_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.oup, :, :]


class FastDetector(nn.Module):
    def __init__(self, num_classes):
        super(FastDetector, self).__init__()

        # Lightweight backbone with Ghost modules
        self.backbone = nn.Sequential(
            GhostModule(3, 16, 3, stride=2),
            GhostModule(16, 32, 3, stride=2),
            GhostModule(32, 64, 3, stride=2),
            GhostModule(64, 128, 3, stride=2),
        )

        # Feature Pyramid Network
        self.fpn = nn.ModuleList(
            [
                DepthwiseSeparableConv(128, 128, 3, 1),
                DepthwiseSeparableConv(128, 64, 3, 1),
                DepthwiseSeparableConv(64, 32, 3, 1),
            ]
        )

        # SE blocks for each FPN level
        self.se_blocks = nn.ModuleList(
            [
                SEBlock(128),
                SEBlock(128),
                SEBlock(64),
                SEBlock(32),
            ]
        )

        # Detection heads
        self.heads = nn.ModuleList(
            [
                self._make_head(128, num_classes),
                self._make_head(128, num_classes),
                self._make_head(64, num_classes),
                self._make_head(32, num_classes),
            ]
        )

    def _make_head(self, in_channels, num_classes):
        return nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels, 3, 1),
            nn.Conv2d(
                in_channels, num_classes + 4, 1
            ),  # cls + x, y, w, h (anchor-free)
        )

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            features.append(x)

        # FPN
        for i in range(len(features) - 1, 0, -1):
            features[i - 1] = features[i - 1] + F.interpolate(
                self.fpn[i - 1](features[i]), size=features[i - 1].shape[2:]
            )

        # Apply SE blocks and get predictions
        outputs = []
        for feature, se_block, head in zip(
            features, self.se_blocks, self.heads
        ):
            feature = se_block(feature)
            outputs.append(head(feature).flatten(start_dim=2))

        return outputs


# Example usage
num_classes = 80  # COCO dataset
model = FastDetector(num_classes)
input_tensor = torch.randn(1, 3, 416, 416)
outputs = model(input_tensor)
for i, output in enumerate(outputs):
    print(f"Output {i} shape:", output.shape)
