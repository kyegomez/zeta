import torch
import torch.nn as nn
import torch.nn.functional as F


class PolymorphicNeuronLayer(nn.Module):
    def __init__(self, in_features, out_features, activation_functions):
        """
        Initialize the Polymorphic Neuron Layer.
        :param in_features: Number of input features.
        :param out_features: Number of output features (neurons).
        :param activation_functions: List of activation functions to choose from.

        Example:
        >>> x = torch.randn(1, 10)
        >>> neuron = PolymorphicNeuronLayer(in_features=10, out_features=5, activation_functions=[F.relu, F.tanh, F.sigmoid])
        >>> output = neuron(x)
        >>> output.shape
        """
        super(PolymorphicNeuronLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_functions = activation_functions
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        """
        Forward pass of the layer.
        :param x: Input tensor.
        :return: Output tensor after applying polymorphic neurons.
        """
        # Linear transformation
        x = F.linear(x, self.weights, self.bias)

        # Apply activation function dynamically
        outputs = []
        for i in range(self.out_features):
            # Example criterion: Use mean of input for selecting activation function
            criterion = x[:, i].mean()
            activation_idx = int(criterion % len(self.activation_functions))
            activation_function = self.activation_functions[activation_idx]
            outputs.append(activation_function(x[:, i]))

        # Stack outputs along the feature dimension
        return torch.stack(outputs, dim=1)


# # Example usage
# polymorphic_layer = PolymorphicNeuronLayer(in_features=10, out_features=5, )

# # Example input
# input_tensor = torch.randn(1, 10)

# # Forward pass
# output = polymorphic_layer(input_tensor)
