from torch import nn
from einops.layers.torch import Rearrange


class SuperResolutionNet(nn.Module):
    """
    Super Resolution Network for MNIST classification.

    Usage:
        net = SuperResolutionNet()
        net(x)



    """

    def __init__(
        self,
        upscale_factor=2,
    ):
        super(SuperResolutionNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, upscale_factor**2, kernel_size=3, padding=1),
            Rearrange(
                "b (h1 w2) h w -> b (h h2) (w w2)",
                h2=upscale_factor,
                w2=upscale_factor,
            ),
        )

    def forward(self, x):
        """Run the forward pass"""
        return self.net(x)
