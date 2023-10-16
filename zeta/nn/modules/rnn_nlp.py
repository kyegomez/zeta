from torch import nn
from einops import rearrange


class RNNL(nn.Module):
    """
    RNN for language

    Args:
    vocab_size: size of vocabulary
    embedding_dim: dimension of embedding
    hidden_dim: dimension of hidden layer
    output_dim: dimension of output
    n_layers: number of layers
    bidirectional: bidirectional

    Usage:
        net = RNNL(
            vocab_size=10,
            embedding_dim=20,
            hidden_dim=50,
            output_dim=10,
            n_layers=2,
            bidirectional=True,
            dropout=0.5,
        )
        net(x)

    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * self.directions, output_dim)

    def forward(self, x):
        """
        Forward pass of the network.
        """
        embedded = self.dropout(self.embedding(x))

        output, (hidden, cell) = self.rnn(embedded)

        hidden = rearrange(
            hidden,
            "(layer dir) b c -> layer b (dir c)",
            dir=self.directions,
        )

        # take the final layers hidden
        return self.fn(self.dropout(hidden[-1]))
