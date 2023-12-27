import torch
import pytest
from zeta.models import MaxVit


# Fixture to create an instance of the MaxVit class.
@pytest.fixture
def maxvit():
    maxvit = MaxVit(
        num_classes=10,
        dim=128,
        depth=(2, 2),
        dim_head=32,
        dim_conv_stem=32,
        window_size=7,
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.01,
        channels=3,
    )
    return maxvit


# Test constructor
def test_maxvit_constructor(maxvit):
    assert maxvit.num_classes == 10
    assert maxvit.dim == 128
    assert maxvit.depth == (2, 2)
    assert maxvit.dim_head == 32
    assert maxvit.dim_conv_stem == 32
    assert maxvit.window_size == 7
    assert maxvit.mbconv_expansion_rate == 4
    assert maxvit.mbconv_shrinkage_rate == 0.25
    assert maxvit.dropout == 0.01
    assert maxvit.channels == 3


# Test `forward` method
def test_forward_returns_correct_shape(maxvit):
    from torch.autograd import Variable

    x = Variable(torch.randn(1, 1, 224, 224))
    result = maxvit.forward(x)
    assert result.size() == (1, 10)


def test_forward_returns_correct_datatype(maxvit):
    from torch.autograd import Variable

    x = Variable(torch.randn(1, 1, 224, 224))
    result = maxvit.forward(x)
    assert isinstance(result, torch.Tensor)
