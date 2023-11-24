import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomMLP(nn.Module):
    """
    A customizable Multi-Layer Perceptron (MLP).

    Attributes:
        layers (nn.ModuleList): List of linear layers.
        activation_fn (callable): Activation function to be applied after each layer.
        dropout (float): Dropout probability for regularization.

    Parameters:
        layer_sizes (list of int): List of layer sizes including input and output layer.
        activation (str, optional): Type of activation function. Default is 'relu'.
        dropout (float, optional): Dropout probability. Default is 0.0 (no dropout).
    """

    def __init__(self, layer_sizes, activation="relu", dropout=0.0):
        super(CustomMLP, self).__init__()

        # Validate input parameters
        if not isinstance(layer_sizes, list) or len(layer_sizes) < 2:
            raise ValueError(
                "layer_sizes must be a list with at least two integers"
                " representing input and output sizes."
            )
        if not all(isinstance(size, int) and size > 0 for size in layer_sizes):
            raise ValueError(
                "All elements in layer_sizes must be positive integers."
            )

        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("dropout must be a float between 0.0 and 1.0")

        # Define the activation function
        if activation == "relu":
            self.activation_fn = F.relu
        elif activation == "sigmoid":
            self.activation_fn = torch.sigmoid
        elif activation == "tanh":
            self.activation_fn = torch.tanh
        else:
            raise ValueError(
                "Unsupported activation function. Supported: 'relu', 'sigmoid',"
                " 'tanh'."
            )

        # Create layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation_fn(x)
            x = self.dropout(x)
        x = self.layers[-1](x)  # No activation or dropout on the last layer
        return x


# Example Usage:
# mlp = CustomMLP(layer_sizes=[10, 5, 2], activation='relu', dropout=0.5)
# input_data = torch.randn(1, 10)
# output = mlp(input_data)
