import torch
import torch.nn as nn


class FlexiConv(nn.Module):
    """
    FlexiConv is an experimental and flexible convolutional layer that adapts to the input data.

    This layer uses parameterized Gaussian functions to weigh the importance of each pixel
    in the receptive field and applies a depthwise separable convolution for efficiency.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0

    Example:
        >>> flexiconv = FlexiConv(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        >>> input_tensor = torch.randn(1, 3, 224, 224)  # Example input batch
        >>> output = flexiconv(input_tensor)
        >>> output.shape
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0
    ):
        super(FlexiConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding

        # Gaussian weights
        self.gaussian_weights = nn.Parameter(
            torch.randn(in_channels, *self.kernel_size)
        )

        # Depthwise separable convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1
        )

        # Initialization of the parameters
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_normal_(
            self.depthwise.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.depthwise.bias, 0)
        nn.init.kaiming_normal_(
            self.pointwise.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.pointwise.bias, 0)
        nn.init.normal_(self.gaussian_weights, mean=0, std=0.1)

    def forward(self, x):
        """
        Forward pass of the FlexiConv layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The result of the flexible convolution.
        """
        # Apply depthwise convolution
        depthwise_out = self.depthwise(x)

        # Generate a Gaussian mask for each channel
        gaussian_mask = torch.exp(-torch.square(self.gaussian_weights))

        # Use einsum to apply the gaussian mask with depthwise convolution output.
        # 'bcij,ckl->bcijkl' denotes a mapping from the batch and channel dimensions (bc),
        # input spatial dimensions (ij), and the kernel dimensions (kl) to a combined output tensor.
        combined = torch.einsum(
            "bcij,ckl->bcijkl", depthwise_out, gaussian_mask
        )

        # Sum over the kernel dimensions to apply the gaussian mask
        weighted_out = combined.sum(dim=-2).sum(dim=-2)

        # Apply pointwise convolution
        out = self.pointwise(weighted_out)

        return out
