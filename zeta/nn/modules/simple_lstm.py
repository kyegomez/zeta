import torch
from torch import nn, Tensor


class SimpleLSTMCell(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        """
        Simple LSTM cell implementation.

        Args:
            dim (int): The input dimension.
            hidden_dim (int): The hidden dimension.
        """
        super(SimpleLSTMCell, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        # Linear layers for input gate, forget gate, output gate, and cell state
        self.W_i = nn.Linear(dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)

        self.W_f = nn.Linear(dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)

        self.W_o = nn.Linear(dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)

        self.W_c = nn.Linear(dim, hidden_dim)
        self.U_c = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> Tensor:
        """
        Forward pass of the Simple LSTM cell.

        Args:
            x (Tensor): The input tensor of shape (batch_size, dim).
            h (Tensor): The previous hidden state tensor of shape (batch_size, hidden_dim).
            c (Tensor): The previous cell state tensor of shape (batch_size, hidden_dim).

        Returns:
            Tensor: The next hidden state tensor.
            Tensor: The next cell state tensor.
        """
        # Compute input gate
        i = torch.sigmoid(self.W_i(x) + self.U_i(h))

        # Compute forget gate
        f = torch.sigmoid(self.W_f(x) + self.U_f(h))

        # Compute output gate
        o = torch.sigmoid(self.W_o(x) + self.U_o(h))

        # Compute new cell candidate
        c_tilde = torch.tanh(self.W_c(x) + self.U_c(h))

        # Update cell state
        c_next = f * c + i * c_tilde

        # Update hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class SimpleLSTM(nn.Module):
    """
    Simple LSTM implementation.

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension.
        depth (int): The number of LSTM layers.
        output_dim (int): The output dimension.
    """

    def __init__(self, dim: int, hidden_dim: int, depth: int, output_dim: int):
        super(SimpleLSTM, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        # LSTM cells
        self.cells = nn.ModuleList(
            [
                SimpleLSTMCell(dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(depth)
            ]
        )

        # Final output layer
        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.sequential = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_length, _ = x.shape

        # Init hidden and cell states with zeros
        h = [
            torch.zeros(batch_size, self.hidden_dim).to(x.device)
            for _ in range(self.depth)
        ]
        c = [
            torch.zeros(batch_size, self.hidden_dim).to(x.device)
            for _ in range(self.depth)
        ]

        # Collect outputs for each time step
        outputs = []

        # Iterate through each time step in the sequence
        for t in range(seq_length):
            # Extract the input for the current time step
            x_t = x[:, t, :]

            # Pass through each LSTM cell
            for layer in range(self.depth):
                h[layer], c[layer] = self.cells[layer](x_t, h[layer], c[layer])
                x_t = h[layer]

            # Collect the output from the final LSTM layer
            outputs.append(h[-1].unsqueeze(1))

        # Concatenate the outputs along the time dimension
        outputs = torch.cat(outputs, dim=1)
        print(outputs.shape)
        b, s, d = outputs.shape

        # Apply the fully connected layer
        # outputs = self.sequential(outputs)
        outputs = nn.Sequential(
            nn.Linear(d, self.dim),
            nn.LayerNorm(self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
            # nn.Softmax(dim=1),
        )(outputs)

        return outputs


# # Example usage:
# if __name__ == "__main__":
#     batch_size = 32
#     seq_length = 10
#     dim = 50
#     hidden_dim = 100
#     num_layers = 2
#     output_dim = 30

#     model = SimpleLSTM(dim, hidden_dim, num_layers, output_dim)
#     inputs = torch.randn(batch_size, seq_length, dim)
#     outputs = model(inputs)
#     print(outputs)  # Expected output shape: (batch_size, seq_length, output_dim)
