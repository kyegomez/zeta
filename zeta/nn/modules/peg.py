from torch import nn, Tensor


class PEG(nn.Module):
    """
    PEG (Positional Encoding Generator) module.

    Args:
        dim (int): The input dimension.
        kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
    """

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.proj = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            stride=1,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the PEG module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        return self.proj(x) + x
