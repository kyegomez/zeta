import torch
import torch.nn as nn
import pytest
from zeta.nn.modules import StochasticSkipBlocK


# Testing instance creation and basic properties
def test_init():
    sb1 = nn.Linear(5, 3)
    block = StochasticSkipBlocK(sb1, p=0.7)
    assert isinstance(block, nn.Module)
    assert block.p == 0.7
    assert block.sb1 == sb1


# Testing forward pass behaviour
def test_forward(monkeypatch):
    sb1 = nn.Linear(5, 3)
    block = StochasticSkipBlocK(sb1, p=0.7)
    x = torch.rand(5)

    # Mock torch.rand() to return 0.8 to test the 'skip' scenario
    def mock_rand(*args, **kwargs):
        return torch.tensor([0.8])

    monkeypatch.setattr(torch, "rand", mock_rand)
    block.training = True
    assert torch.allclose(block.forward(x), x)

    # Mock torch.rand() to return 0.6 to test the 'non-skip' scenario
    def mock_rand_2(*args, **kwargs):
        return torch.tensor([0.6])

    monkeypatch.setattr(torch, "rand", mock_rand_2)
    assert not torch.allclose(block.forward(x), x)


# Testing invalid input handling
def test_invalid_p_constructor():
    sb1 = nn.Linear(5, 3)

    with pytest.raises(ValueError):
        # p value less than 0
        _ = StochasticSkipBlocK(sb1, p=-0.1)

    with pytest.raises(ValueError):
        # p value more than 1
        _ = StochasticSkipBlocK(sb1, p=1.1)
