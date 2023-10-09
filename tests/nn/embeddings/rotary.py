import pytest
import torch
from zeta.nn.embeddings.rope import RotaryEmbedding


def test_rotaryembedding_initialization():
    model = RotaryEmbedding(dim=512)
    assert isinstance(model, RotaryEmbedding)
    assert model.inv_freq.shape == (256,)
    assert model.interpolation_factor == 1.0


def test_rotaryembedding_forward():
    model = RotaryEmbedding(dim=512)
    freqs, scale = model(10, device="cpu")
    assert freqs.shape == (10, 512)
    assert scale == 1.0


@pytest.mark.parametrize("seq_len", [0])
def test_rotaryembedding_forward_edge_cases(seq_len):
    model = RotaryEmbedding(dim=512)
    with pytest.raises(Exception):
        model(seq_len, device="cpu")


def test_rotaryembedding_forward_invalid_dimensions():
    model = RotaryEmbedding(dim=512)
    with pytest.raises(Exception):
        model(10, device="cuda")
