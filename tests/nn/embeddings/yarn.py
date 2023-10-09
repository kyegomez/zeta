import pytest
import torch
from zeta.nn.embeddings.yarn import YarnEmbedding

def test_yarnembedding_initialization():
    model = YarnEmbedding(dim=512)
    assert isinstance(model, YarnEmbedding)
    assert model.dim == 512
    assert model.max_position_embeddings == 2048
    assert model.base == 10000

def test_yarnembedding_forward():
    model = YarnEmbedding(dim=512)
    x = torch.randn(1, 10, 512)
    cos_cached, sin_cached = model(x, seq_len=10)
    assert cos_cached.shape == (1, 1, 10, 512)
    assert sin_cached.shape == (1, 1, 10, 512)

@pytest.mark.parametrize("seq_len", [0])
def test_yarnembedding_forward_edge_cases(seq_len):
    model = YarnEmbedding(dim=512)
    x = torch.randn(1, seq_len, 512)
    with pytest.raises(Exception):
        model(x, seq_len=seq_len)

def test_yarnembedding_forward_invalid_dimensions():
    model = YarnEmbedding(dim=512)
    x = torch.randn(1, 10, 256)
    with pytest.raises(Exception):
        model(x, seq_len=10)