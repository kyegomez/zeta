import pytest
import torch
from zeta.nn.embeddings.qfsp_embeddings import QFTSPEmbedding


def test_qsembeddings_init():
    vocab_size = 10000
    dim = 512
    model = QFTSPEmbedding(vocab_size, dim)
    assert model.embed_dim == dim
    assert model.base_embeddings.num_embeddings == vocab_size
    assert model.superposed_embeddings.num_embeddings == vocab_size


def test_qsembeddings_forward_weighted_sum():
    vocab_size = 10000
    dim = 512
    model = QFTSPEmbedding(vocab_size, dim)
    x = torch.randint(0, vocab_size, (1, 10))
    context_vector = torch.rand(1, 10)
    embeddings = model(x, context_vector, "weighted_sum")
    assert embeddings.shape == (1, 10, dim)


def test_qsembeddings_forward_dot_product():
    vocab_size = 10000
    dim = 512
    model = QFTSPEmbedding(vocab_size, dim)
    x = torch.randint(0, vocab_size, (1, 10))
    context_vector = torch.rand(1, 10)
    embeddings = model(x, context_vector, "dot_product")
    assert embeddings.shape == (1, 10, dim)


def test_qsembeddings_forward_cosine_similarity():
    vocab_size = 10000
    dim = 512
    model = QFTSPEmbedding(vocab_size, dim)
    x = torch.randint(0, vocab_size, (1, 10))
    context_vector = torch.rand(1, 10)
    embeddings = model(x, context_vector, "cosine_similarity")
    assert embeddings.shape == (1, 10, dim)


def test_qsembeddings_forward_gated():
    vocab_size = 10000
    dim = 512
    model = QFTSPEmbedding(vocab_size, dim)
    x = torch.randint(0, vocab_size, (1, 10))
    context_vector = torch.rand(1, 10)
    embeddings = model(x, context_vector, "gated")
    assert embeddings.shape == (1, 10, dim)


def test_qsembeddings_forward_concat_linear():
    vocab_size = 10000
    dim = 512
    model = QFTSPEmbedding(vocab_size, dim)
    x = torch.randint(0, vocab_size, (1, 10))
    context_vector = torch.rand(1, 10)
    embeddings = model(x, context_vector, "concat_linear")
    assert embeddings.shape == (1, 10, dim)


def test_qsembeddings_forward_invalid_mode():
    vocab_size = 10000
    dim = 512
    model = QFTSPEmbedding(vocab_size, dim)
    x = torch.randint(0, vocab_size, (1, 10))
    context_vector = torch.rand(1, 10)
    with pytest.raises(ValueError):
        model(x, context_vector, "invalid_mode")


def test_qsembeddings_forward_large_input():
    vocab_size = 10000
    dim = 512
    model = QFTSPEmbedding(vocab_size, dim)
    x = torch.randint(0, vocab_size, (1000, 1000))
    context_vector = torch.rand(1000, 1000)
    embeddings = model(x, context_vector, "weighted_sum")
    assert embeddings.shape == (1000, 1000, dim)


def test_qsembeddings_forward_large_dim():
    vocab_size = 10000
    dim = 10000
    model = QFTSPEmbedding(vocab_size, dim)
    x = torch.randint(0, vocab_size, (1, 10))
    context_vector = torch.rand(1, 10)
    embeddings = model(x, context_vector, "weighted_sum")
    assert embeddings.shape == (1, 10, dim)


def test_qsembeddings_forward_large_vocab_size():
    vocab_size = 1000000
    dim = 512
    model = QFTSPEmbedding(vocab_size, dim)
    x = torch.randint(0, vocab_size, (1, 10))
    context_vector = torch.rand(1, 10)
    embeddings = model(x, context_vector, "weighted_sum")
    assert embeddings.shape == (1, 10, dim)
