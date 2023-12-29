import torch

from zeta.nn.modules.lora import Lora


def test_lora_forward():
    lora = Lora(10, 10)
    x = torch.randn(1, 10)
    output = lora.forward(x)
    assert output.shape == (1, 10)
    assert torch.allclose(output, x @ lora.weight)


def test_lora_forward_zero_input():
    lora = Lora(10, 10)
    x = torch.zeros(1, 10)
    output = lora.forward(x)
    assert output.shape == (1, 10)
    assert torch.all(output == 0)


def test_lora_forward_one_input():
    lora = Lora(10, 10)
    x = torch.ones(1, 10)
    output = lora.forward(x)
    assert output.shape == (1, 10)
    assert torch.allclose(output, x @ lora.weight)
