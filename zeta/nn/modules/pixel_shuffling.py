from torch import nn, Tensor


class PixelShuffleDownscale(nn.Module):
    def __init__(self, downscale_factor: int = 2):
        """
        Initializes a PixelShuffleDownscale module.

        Args:
            downscale_factor (int): The factor by which the input will be downscaled.

        Example:
        >>> downscale_factor = 2
        >>> model = PixelShuffleDownscale(downscale_factor)
        >>> input_tensor = torch.rand(1, 256, 448, 448)
        >>> output_tensor = model(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 64, 896, 896])
        """
        super(PixelShuffleDownscale, self).__init__()
        self.downscale_factor = downscale_factor
        # Initialize the pixel shuffle with an upscale factor which will actually be used to downscale
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=downscale_factor)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass of the PixelShuffleDownscale module.

        Args:
            x (torch.Tensor): The input tensor with shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: The output tensor after downsampling using pixel shuffle.
        """
        # x should have a shape of [batch_size, channels, height, width]
        # We first need to adapt the number of channels so that pixel shuffle can be applied
        batch_size, channels, height, width = x.shape
        new_channels = channels // (self.downscale_factor**2)
        if new_channels * (self.downscale_factor**2) != channels:
            raise ValueError(
                "The number of channels must be divisible by"
                " (downscale_factor^2)"
            )

        # Reshape x to the shape expected by pixel shuffle
        x = x.reshape(
            batch_size, new_channels, self.downscale_factor**2, height, width
        )
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(
            batch_size,
            new_channels * (self.downscale_factor**2),
            height,
            width,
        )

        # Apply pixel shuffle to reduce spatial dimensions and increase channel depth
        x = self.pixel_shuffle(x)

        return x


# # Example of usage
# downscale_factor = (
#     2  # This factor needs to be determined based on the required reduction
# )
# model = PixelShuffleDownscale(downscale_factor)
# input_tensor = torch.rand(1, 256, 448, 448)  # Example input tensor
# output_tensor = model(input_tensor)
# print(output_tensor.shape)  # This will print the shape of the output tensor
