import pytest
import torch
from zeta.nn.modules.token_learner import TokenLearner
from torch import nn


def test_tokenlearner_initialization():
    model = TokenLearner(dim=256, num_output_tokens=8)
    assert isinstance(model, TokenLearner)
    assert model.num_output_tokens == 8
    assert isinstance(model.net, nn.Sequential)


def test_tokenlearner_forward():
    model = TokenLearner(dim=256, num_output_tokens=8)
    x = torch.randn(1, 256, 10, 10)
    output = model(x)
    assert output.shape == (1, 256, 8)


@pytest.mark.parametrize("dim", [0])
def test_tokenlearner_forward_edge_cases(dim):
    model = TokenLearner(dim=dim, num_output_tokens=8)
    x = torch.randn(1, dim, 10, 10)
    with pytest.raises(Exception):
        model(x)


def test_tokenlearner_forward_invalid_dimensions():
    model = TokenLearner(dim=256, num_output_tokens=8)
    x = torch.randn(1, 128, 10, 10)
    with pytest.raises(Exception):
        model(x)
