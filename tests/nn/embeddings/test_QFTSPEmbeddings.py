import pytest
import torch
from zeta.nn.embeddings.qft_embeddings import QFTSPEmbeddings


def test_qftspembeddings_init():
    vocab_size = 10000
    dim = 512
    model = QFTSPEmbeddings(vocab_size, dim)
    assert model.vocab_size == vocab_size
    assert model.dim == dim


def test_qftspembeddings_forward():
    vocab_size = 10000
    dim = 512
    model = QFTSPEmbeddings(vocab_size, dim)
    x = torch.randint(0, vocab_size, (1, 10))
    embeddings = model(x)
    assert embeddings.shape == (1, 10, dim)


def test_qftspembeddings_forward_zero_dim():
    vocab_size = 10000
    dim = 0
    model = QFTSPEmbeddings(vocab_size, dim)
    x = torch.randint(0, vocab_size, (1, 10))
    embeddings = model(x)
    assert embeddings.shape == (1, 10, 0)


def test_qftspembeddings_forward_odd_dim():
    vocab_size = 10000
    dim = 513
    model = QFTSPEmbeddings(vocab_size, dim)
    x = torch.randint(0, vocab_size, (1, 10))
    embeddings = model(x)
    assert embeddings.shape == (1, 10, dim)


def test_qftspembeddings_forward_large_input():
    vocab_size = 10000
    dim = 512
    model = QFTSPEmbeddings(vocab_size, dim)
    x = torch.randint(0, vocab_size, (1000, 1000))
    embeddings = model(x)
    assert embeddings.shape == (1000, 1000, dim)


def test_qftspembeddings_forward_large_dim():
    vocab_size = 10000
    dim = 10000
    model = QFTSPEmbeddings(vocab_size, dim)
    x = torch.randint(0, vocab_size, (1, 10))
    embeddings = model(x)
    assert embeddings.shape == (1, 10, dim)


def test_qftspembeddings_forward_large_vocab_size():
    vocab_size = 1000000
    dim = 512
    model = QFTSPEmbeddings(vocab_size, dim)
    x = torch.randint(0, vocab_size, (1, 10))
    embeddings = model(x)
    assert embeddings.shape == (1, 10, dim)


def test_qftspembeddings_forward_negative_dim():
    vocab_size = 10000
    dim = -512
    with pytest.raises(ValueError):
        model = QFTSPEmbeddings(vocab_size, dim)


def test_qftspembeddings_forward_negative_vocab_size():
    vocab_size = -10000
    dim = 512
    with pytest.raises(ValueError):
        model = QFTSPEmbeddings(vocab_size, dim)


def test_qftspembeddings_forward_zero_vocab_size():
    vocab_size = 0
    dim = 512
    with pytest.raises(ValueError):
        model = QFTSPEmbeddings(vocab_size, dim)
