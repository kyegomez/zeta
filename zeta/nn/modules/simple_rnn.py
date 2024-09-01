# replace some of the activation functions from sigmoid to exponential function - e ^ x
# Memory saving: make the memory larger --> associate memory --> increase


from torch import nn, Tensor


class SimpleRNN(nn.Module):
    """
    A simple recurrent neural network module.

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The dimension of the hidden state.
    """

    def __init__(
        self,
        dim: int = None,
        hidden_dim: int = None,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.act = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the simple RNN module.

        Args:
            x (Tensor): The input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Tensor: The output tensor of shape (batch_size, sequence_length, hidden_dim).
        """
        b, s, d = x.shape

        h = self.act(x)

        return h
