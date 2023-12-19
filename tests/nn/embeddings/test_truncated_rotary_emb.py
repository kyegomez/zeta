import pytest
from zeta.nn.embeddings.truncated_rope import TruncatedRotaryEmbedding


# Test case for default initialization
def test_default_init():
    dim = 10
    a = 0.5
    b = 1.0
    rho = 0.0
    module = TruncatedRotaryEmbedding(dim, a, b, rho)
    assert module.dim == dim
    assert module.a == a
    assert module.b == b
    assert module.rho == rho


# Test case for forward pass
def test_forward_pass():
    dim = 10
    a = 0.5
    b = 1.0
    rho = 0.0
    module = TruncatedRotaryEmbedding(dim, a, b, rho)
    seq_len = 10
    device = "cpu"
    result = module(seq_len, device)
    assert result.shape == (seq_len, dim)


# Test case for forward pass with a different device
def test_forward_pass_device():
    dim = 10
    a = 0.5
    b = 1.0
    rho = 0.0
    module = TruncatedRotaryEmbedding(dim, a, b, rho)
    seq_len = 10
    device = "cuda"
    result = module(seq_len, device)
    assert result.device == device


# Test case for initializing with negative dimension
def test_negative_dimension():
    dim = -10
    a = 0.5
    b = 1.0
    rho = 0.0
    with pytest.raises(ValueError):
        TruncatedRotaryEmbedding(dim, a, b, rho)


# Test case for initializing with a > b
def test_a_greater_than_b():
    dim = 10
    a = 1.0
    b = 0.5
    rho = 0.0
    with pytest.raises(ValueError):
        TruncatedRotaryEmbedding(dim, a, b, rho)


# Test case for initializing with rho > b
def test_rho_greater_than_b():
    dim = 10
    a = 0.5
    b = 1.0
    rho = 1.5
    with pytest.raises(ValueError):
        TruncatedRotaryEmbedding(dim, a, b, rho)
