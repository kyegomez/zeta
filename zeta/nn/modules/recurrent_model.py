from einops import rearrange
from torch import nn


class RNN(nn.Module):
    """
    Recurrent Neural Network for MNIST classification.

    Usage:
        net = RNN(
            ntoken=10,
            ninp=20,
            nhid=50,
            nlayers=2,
            dropout=0.5,
        )
        net(x)

    """

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNN, self).__init__()

        self.drop = nn.Dropout(p=dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

    def forward(self, input, hidden):
        """
        Take a forward pass through the network.

        Usage:
            net = RNN()
            net(x)
        """
        t, b = input.shape
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = rearrange(
            self.drop(output),
            "t b nhid -> (t b) nhid",
        )
        decoded = rearrange(
            self.decoder(output), "(t b) token -> t b token", t=t, b=b
        )
        return decoded, hidden
