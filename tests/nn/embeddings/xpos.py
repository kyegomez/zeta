import pytest
import torch
from torch import nn
from zeta.nn.embeddings.xpos_relative_position import XPOS

def test_xpos_initialization():
    model = XPOS(head_dim=256)
    assert isinstance(model, XPOS)
    assert model.head_dim == 256
    assert model.scale_base == 512
    assert model.scale.shape == (128,)

def test_xpos_forward():
    model = XPOS(head_dim=256)
    x = torch.randn(1, 10, 256)
    output = model(x)
    assert output.shape == (1, 10, 256)

@pytest.mark.parametrize("head_dim", [0])
def test_xpos_forward_edge_cases(head_dim):
    model = XPOS(head_dim=head_dim)
    x = torch.randn(1, 10, head_dim)
    with pytest.raises(Exception):
        model(x)

def test_xpos_forward_invalid_dimensions():
    model = XPOS(head_dim=256)
    x = torch.randn(1, 10, 128)
    with pytest.raises(Exception):
        model(x)