from torch import nn


class DualPathBlock(nn.Module):
    def __init__(self, submodule1, submodule2):
        """
        DualPathBlock is a module that combines the output of two submodules by element-wise addition.

        Args:
            submodule1 (nn.Module): The first submodule.
            submodule2 (nn.Module): The second submodule.
        """
        super().__init__()
        self.submodule1 = submodule1
        self.submodule2 = submodule2

    def forward(self, x):
        """
        Forward pass of the DualPathBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor obtained by adding the outputs of submodule1 and submodule2.
        """
        return self.submodule1(x) + self.submodule2(x)
