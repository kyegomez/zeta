# PytorchGELUTanh

import pytest
import torch
from torch import nn

from zeta.nn import PytorchGELUTanh


def test_PytorchGELUTanh_initialization_success():
    model = PytorchGELUTanh()
    assert isinstance(model, nn.Module)


@pytest.mark.parametrize("torch_version", ["1.11.0", "1.11.9"])
def test_PytorchGELUTanh_initialization_fails_with_old_pytorch(
    monkeypatch, torch_version
):
    monkeypatch.setattr(torch, "__version__", torch_version)
    with pytest.raises(ImportError) as e_info:
        PytorchGELUTanh()
    assert (
        str(e_info.value)
        == f"You are using torch=={torch.__version__}, but torch>=1.12.0 is"
        " required to use PytorchGELUTanh. Please upgrade torch."
    )


def test_PytorchGELUTanh_forward_propagation():
    tensor_input = torch.Tensor([2.0, 3.0, 4.0])
    model = PytorchGELUTanh()
    output = model.forward(tensor_input)
    target = nn.functional.gelu(tensor_input, approximate="tanh")
    assert torch.allclose(output, target)


def test_PytorchGELUTanh_with_random_inputs():
    tensor_input = torch.rand(10, 10)
    model = PytorchGELUTanh()
    output = model.forward(tensor_input)
    target = nn.functional.gelu(tensor_input, approximate="tanh")
    assert torch.allclose(output, target)
