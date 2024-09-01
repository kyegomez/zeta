from einops import rearrange, reduce
from torch import nn


class CNNNew(nn.Module):
    """
    CNN for language

    Args:
    vocab_size: size of vocabulary
    embedding_dim: dimension of embedding
    n_filters: number of filters
    filter_sizes: filter sizes
    output_dim: dimension of output
    dropout: dropout rate

    Usage:
        net = CNNNew(
            vocab_size=10,
            embedding_dim=20,
            n_filters=100,
            filter_sizes=[3, 4, 5],
            output_dim=10,
            dropout=0.5,
        )
        net(x)

    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        output_dim,
        dropout,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(embedding_dim, n_filters, kernel_size=size)
                for size in filter_sizes
            ]
        )
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of CNNNew

        """
        x = rearrange(x, "b t -> b t")
        emb = rearrange(self.embedding(x), "t b c -> b c t")
        pooled = [
            reduce(conv(emb), "b c t -> b c", "max") for conv in self.convs
        ]
        concatenated = rearrange(pooled, "filter b c -> b (filter c)")
        return self.fc(self.dropout(concatenated))
