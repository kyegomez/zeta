from torch import nn
from einops.layers.torch import Rearrange, Reduce


def FastTextNew(vocab_size, embedding_dim, output_dim):
    """
    FastText for language

    Args:
    vocab_size: size of vocabulary
    embedding_dim: dimension of embedding
    output_dim: dimension of output

    Usage:
        net = FastTextNew(
            vocab_size=10,
            embedding_dim=20,
            output_dim=10,
        )
        net(x)

    """
    return nn.Sequential(
        Rearrange("t b -> t b "),
        nn.Embedding(vocab_size, embedding_dim),
        Reduce("t b c -> b c", "mean"),
        nn.Linear(embedding_dim, output_dim),
        Rearrange("b c -> b c"),
    )
