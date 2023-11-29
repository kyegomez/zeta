import torch
import torch.nn as nn


class PolymorphicActivation(nn.Module):
    """
    A Polymorphic Activation Function in PyTorch.

    This activation function combines aspects of sigmoid and tanh functions,
    controlled by a learnable parameter alpha. The behavior of the function
    adapts based on the input and the state of alpha during training.

    Attributes:
    -----------
    alpha : torch.nn.Parameter
        A trainable parameter that modulates the behavior of the activation function.

    Methods:
    --------
    forward(x):
        Computes the polymorphic activation function on the input tensor x.

    Examples:
    # Create an instance of the activation function
    poly_act = PolymorphicActivation(initial_alpha=0.8)

    # Example input tensor
    input_tensor = torch.randn(5)

    # Apply the polymorphic activation function
    output = poly_act(input_tensor)
    output

    """

    def __init__(self, initial_alpha: float = 0.5):
        """
        Initializes the PolymorphicActivation module.

        Parameters:
        -----------
        initial_alpha : float (optional)
            The initial value of the alpha parameter. Defaults to 0.5.
        """
        super(PolymorphicActivation, self).__init__()
        if not isinstance(initial_alpha, float):
            raise TypeError("initial_alpha must be a float.")
        self.alpha = nn.Parameter(torch.tensor([initial_alpha]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Polymorphic Activation Function.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor to the activation function.

        Returns:
        --------
        torch.Tensor
            The result of applying the polymorphic activation function to x.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")

        sigmoid_part = torch.sigmoid(self.alpha * x)
        tanh_part = torch.tanh(x)
        return sigmoid_part + self.alpha * tanh_part
