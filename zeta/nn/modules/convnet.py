from torch import nn

from einops.layers.torch import Rearrange


class ConvNet(nn.Module):
    """
    Convolutional Neural Network for MNIST classification.

    Usage:
        net = ConvNet()
        net(x)

    """

    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv_net_new = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout2d(),
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        """
        Forward pass of the network.
        """
        return self.conv_net_new(x)
