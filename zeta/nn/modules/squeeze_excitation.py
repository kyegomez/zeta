from torch import nn


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
