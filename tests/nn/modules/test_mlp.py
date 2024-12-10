import pytest
import torch
from torch import nn

from zeta.nn.modules.mlp import MLP


def test_mlp_initialization():
    model = MLP(dim_in=256, dim_out=10)
    assert isinstance(model, MLP)
    assert len(model.net) == 3
    assert isinstance(model.net[0], nn.Sequential)
    assert isinstance(model.net[1], nn.Sequential)
    assert isinstance(model.net[2], nn.Linear)


def test_mlp_forward():
    model = MLP(dim_in=256, dim_out=10)
    x = torch.randn(32, 256)
    output = model(x)
    assert output.shape == (32, 10)


@pytest.mark.parametrize("dim_in", [0])
def test_mlp_forward_edge_cases(dim_in):
    model = MLP(dim_in=dim_in, dim_out=10)
    x = torch.randn(32, dim_in)
    with pytest.raises(Exception):
        model(x)


def test_mlp_forward_invalid_dimensions():
    model = MLP(dim_in=256, dim_out=10)
    x = torch.randn(32, 128)
    with pytest.raises(Exception):
        model(x)
