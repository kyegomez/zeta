import pytest
import torch
from zeta.nn.embeddings.abc_pos_emb import AbsolutePositionalEmbedding


def test_absolutepositionalembedding_initialization():
    model = AbsolutePositionalEmbedding(dim=512, max_seq_len=1000)
    assert isinstance(model, AbsolutePositionalEmbedding)
    assert model.scale == 512**-0.5
    assert model.max_seq_len == 1000
    assert model.l2norm_embed == False
    assert model.emb.weight.shape == (1000, 512)


def test_absolutepositionalembedding_forward():
    model = AbsolutePositionalEmbedding(dim=512, max_seq_len=1000)
    x = torch.randn(1, 10, 512)
    output = model(x)
    assert output.shape == (10, 512)


@pytest.mark.parametrize("seq_len", [1001])
def test_absolutepositionalembedding_forward_edge_cases(seq_len):
    model = AbsolutePositionalEmbedding(dim=512, max_seq_len=1000)
    x = torch.randn(1, seq_len, 512)
    with pytest.raises(Exception):
        model(x)


def test_absolutepositionalembedding_forward_invalid_dimensions():
    model = AbsolutePositionalEmbedding(dim=512, max_seq_len=1000)
    x = torch.randn(1, 10, 256)
    with pytest.raises(Exception):
        model(x)
