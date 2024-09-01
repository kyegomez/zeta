from torch import nn, Tensor
from zeta.nn.modules.feedforward import FeedForward


class MRAdapter(nn.Module):
    """
    Multi-Resolution Adapter module for neural networks.

    Args:
        dim (int): The input dimension.
        heads (int, optional): The number of attention heads. Defaults to 8.
        channels (int, optional): The number of channels. Defaults to 64.

    References:
        https://arxiv.org/pdf/2403.03003.pdf
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        channels: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.channels = channels

        # FeedForward
        self.ff = FeedForward(
            dim,
            dim,
            mult=4,
            swish=True,
            post_act_ln=True,
        )

        # Gate
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        # Conv1d
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
        )

    def forward(self, x: Tensor, y: Tensor):
        """
        Forward pass of the MRAdapter module.

        Args:
            x (Tensor): The input tensor.
            y (Tensor): The tensor to be adapted.

        Returns:
            Tensor: The adapted tensor.
        """
        y_skip = y

        x = self.ff(x)

        y = self.conv(y)

        # Gate
        gate = self.gate(x + y)

        # Fusion
        return gate + y + y_skip
