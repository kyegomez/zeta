# MishActivation

import torch
from torch import nn

from zeta.nn import MishActivation


def test_MishActivation_init():
    mish_activation = MishActivation()
    assert mish_activation.act == nn.functional.mish


def test__mish_python():
    mish_activation = MishActivation()
    input = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    expected_output = input * torch.tanh(nn.functional.softplus(input))

    assert torch.equal(mish_activation._mish_python(input), expected_output)


def test_forward():
    mish_activation = MishActivation()
    input = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    expected_output = nn.functional.mish(input)

    assert torch.equal(mish_activation.forward(input), expected_output)
