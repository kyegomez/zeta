import torch.nn as nn


class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth=3):
        """
        Initialize a Fractal Block.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param depth: Depth of the fractal block.
        """
        super(FractalBlock, self).__init__()
        self.depth = depth

        # Base case for recursion
        if depth == 1:
            self.block = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1
            )
        else:
            # Recursive case: create smaller fractal blocks
            self.block1 = FractalBlock(in_channels, out_channels, depth - 1)
            self.block2 = FractalBlock(in_channels, out_channels, depth - 1)

    def forward(self, x):
        """
        Forward pass of the fractal block.
        :param x: Input tensor.
        :return: Output tensor.
        """
        if self.depth == 1:
            return self.block(x)
        else:
            # Recursively compute the outputs of the sub-blocks
            out1 = self.block1(x)
            out2 = self.block2(x)

            # Combine the outputs of the sub-blocks
            return out1 + out2


class FractalNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, block_depth):
        """
        Initialize the Fractal Network.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param num_blocks: Number of fractal blocks in the network.
        :param block_depth: Depth of each fractal block.
        """
        super(FractalNetwork, self).__init__()
        self.blocks = nn.ModuleList(
            [
                FractalBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    block_depth,
                )
                for i in range(num_blocks)
            ]
        )
        self.final_layer = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the fractal network.
        :param x: Input tensor.
        :return: Output tensor.
        """
        for block in self.blocks:
            x = block(x)
        return self.final_layer(x)


# # Example usage
# fractal_net = FractalNetwork(in_channels=3, out_channels=16, num_blocks=4, block_depth=3)

# # Example input
# input_tensor = torch.randn(1, 3, 64, 64)

# # Forward pass
# output = fractal_net(input_tensor)
# print(output)
