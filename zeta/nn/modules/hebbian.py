import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicHebbianGRUModel(nn.Module):
    """
    A basic Hebbian learning model combined with a GRU for text-based tasks.

    This model applies a simple Hebbian update rule to the weights and uses a GRU
    layer for handling sequential data. The ReLU activation function is used for
    introducing non-linearity.

    Parameters:
    - input_dim: Dimension of the input features.
    - hidden_dim: Dimension of the hidden state in the GRU.
    - output_dim: Dimension of the output features.

    The model processes input through the Hebbian updated weights, then through the
    GRU, and finally applies a ReLU activation.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes the Basic Hebbian GRU model.

        Args:
        - input_dim: Dimension of the input features.
        - hidden_dim: Dimension of the hidden state in the GRU.
        - output_dim: Dimension of the output features.
        """
        super(BasicHebbianGRUModel, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
        - x: Input tensor of shape (B, Seqlen, input_dim)

        Returns:
        - Output tensor of shape (B, Seqlen, output_dim)
        """
        # Apply Hebbian updated weights
        x = torch.matmul(x, self.weights)

        # GRU processing
        x, _ = self.gru(x)

        # Apply ReLU activation function
        x = F.relu(x)

        # Final fully connected layer
        x = self.fc(x)
        return x


# # Example usage
input_dim = 512  # Dimension of the input features
hidden_dim = 256  # Dimension of the hidden state in the GRU
output_dim = 128  # Dimension of the output features
model = BasicHebbianGRUModel(input_dim, hidden_dim, output_dim)

x = torch.randn(1, 512, 512)
output = model(x)
print(output.shape)
