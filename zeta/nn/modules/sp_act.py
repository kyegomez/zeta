import torch
from torch import nn


class SPAct(nn.Module):
    def __init__(self, alpha: float = 0.5):
        """
        Initializes the SPAct module.

        Args:
            alpha (float): The weight parameter for the linear combination of the input and the hyperbolic tangent output.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        """
        Performs the forward pass of the SPAct module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the SPAct function.
        """
        return self.alpha * x + (1 - self.alpha) * torch.tanh(x)


# x = torch.randn(1, 3)

# model = SPAct()

# out = model(x)
# print(out)
