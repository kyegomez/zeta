from torch import nn
import pytest
import torch
from zeta.structs import LocalTransformer
from torch.autograd import gradcheck
from zeta.nn import DynamicPositionBias


@pytest.fixture
def transformer():
    return LocalTransformer(
        num_tokens=5000,
        max_seq_len=200,
        dim=128,
        depth=10,
        causal=True,
        local_attn_window_size=50,
        dim_head=32,
        heads=4,
        ff_mult=2,
        attn_dropout=0.1,
        ff_dropout=0.1,
        ignore_index=-1,
        use_xpos=True,
        xpos_scale_base=100,
        use_dynamic_pos_bias=True,
    )


def test_initialization(transformer):
    assert isinstance(transformer, LocalTransformer)
    assert transformer.token_emb.num_embeddings == 5000
    assert transformer.token_emb.embedding_dim == 128
    assert transformer.pos_emb.num_embeddings == 200
    assert transformer.pos_emb.embedding_dim == 128
    assert transformer.max_seq_len == 200
    assert isinstance(transformer.layers, nn.ModuleList)
    assert transformer.local_attn_window_size == 50
    assert isinstance(transformer.dynamic_pos_bias, DynamicPositionBias)
    assert transformer.ignore_index == -1
    assert isinstance(transformer.to_logits, nn.Sequential)


def test_forward(transformer):
    x = torch.rand(10, 250)
    output = transformer.forward(x)
    assert output.shape == torch.Size([10, 250, 5000])


def test_generate(transformer):
    prime = torch.rand(10, 100)
    output = transformer.generate(
        prime, seq_len=50, temperature=0.9, filter_thres=0.8
    )
    assert output.shape == torch.Size([10, 150])


def test_forward_with_loss(transformer):
    x = torch.rand(10, 250)
    loss = transformer.forward(x, return_loss=True)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()


def test_gradient(transformer):
    x = torch.randn(20, 128, dtype=torch.float64, requires_grad=True)
    test = gradcheck(transformer.forward, (x,), eps=1e-6, atol=1e-4)
    assert test


def test_mocking_used_libraries(mocker):
    mock = mocker.patch("torch.nn.Embedding", return_value="Mocked_Embedding")
    transformer = LocalTransformer(
        num_tokens=5000, max_seq_len=200, dim=128, depth=10, causal=True
    )
    transformer.token_emb = mock
    assert transformer.token_emb() == "Mocked_Embedding"
