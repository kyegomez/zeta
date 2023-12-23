from torch import nn

class SimpleResBlock(nn.Module):
    """
    A simple residual block module.

    Args:
        channels (int): The number of input and output channels.

    Attributes:
        pre_norm (nn.LayerNorm): Layer normalization module applied before the projection.
        proj (nn.Sequential): Sequential module consisting of linear layers and GELU activation.

    """

    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        """
        Forward pass of the simple residual block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the residual block.

        """
        x = self.pre_norm(x)
        return x + self.proj(x)
