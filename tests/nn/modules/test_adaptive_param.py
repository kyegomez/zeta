import pytest
import torch
from torch import nn
from zeta.nn.modules.adaptive_parameter_list import AdaptiveParameterList


def test_adaptiveparameterlist_initialization():
    model = AdaptiveParameterList([nn.Parameter(torch.randn(10, 10))])
    assert isinstance(model, AdaptiveParameterList)
    assert len(model) == 1


def test_adaptiveparameterlist_adapt():
    model = AdaptiveParameterList([nn.Parameter(torch.randn(10, 10))])
    model.adapt({0: lambda x: x * 0.9})
    assert torch.allclose(model[0], torch.randn(10, 10) * 0.9, atol=1e-4)


@pytest.mark.parametrize("adaptation_functions", [lambda x: x * 0.9])
def test_adaptiveparameterlist_adapt_edge_cases(adaptation_functions):
    model = AdaptiveParameterList([nn.Parameter(torch.randn(10, 10))])
    with pytest.raises(Exception):
        model.adapt(adaptation_functions)


def test_adaptiveparameterlist_adapt_invalid_dimensions():
    model = AdaptiveParameterList([nn.Parameter(torch.randn(10, 10))])
    with pytest.raises(Exception):
        model.adapt({0: lambda x: x.view(-1)})
