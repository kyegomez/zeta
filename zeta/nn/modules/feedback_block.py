import torch
from torch import nn


class FeedbackBlock(nn.Module):
    def __init__(self, submodule):
        """
        Initializes a FeedbackBlock module.

        Args:
            submodule (nn.Module): The submodule to be used within the FeedbackBlock.
        """
        super().__init__()
        self.submodule = submodule

    def forward(self, x: torch.Tensor, feedback, *args, **kwargs):
        """
        Performs a forward pass through the FeedbackBlock.

        Args:
            x (torch.Tensor): The input tensor.
            feedback: The feedback tensor.
            *args: Additional positional arguments to be passed to the submodule's forward method.
            **kwargs: Additional keyword arguments to be passed to the submodule's forward method.

        Returns:
            torch.Tensor: The output tensor after passing through the FeedbackBlock.
        """
        if feedback is not None:
            x = x + feedback
        return self.submodule(x, *args, **kwargs)
