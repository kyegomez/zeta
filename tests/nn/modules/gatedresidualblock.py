import pytest
import torch
import torch.nn as nn
from torch.autograd import gradcheck
from zeta.nn.modules import GatedResidualBlock


class TestGatedResidualBlock:
    @pytest.fixture(scope="class")
    def init_grb(self):
        sb1 = nn.Linear(3, 3)
        gate_module = nn.Linear(3, 3)
        return GatedResidualBlock(sb1, gate_module)

    # Test instance creation and types
    def test_instance(self, init_grb):
        assert isinstance(init_grb, GatedResidualBlock)
        assert isinstance(init_grb.sb1, nn.Module)
        assert isinstance(init_grb.gate_module, nn.Module)

    # Test forward pass
    def test_forward(self, init_grb):
        x = torch.rand(1, 3)
        out = init_grb(x)
        assert isinstance(out, torch.Tensor)
        assert (
            out.shape == x.shape
        )  # outputs and input tensors should have same shape

    # Test learnable parameters
    def test_parameters(self, init_grb):
        for param in init_grb.parameters():
            assert param.requires_grad

    # Gradients check
    def test_gradients(self, init_grb):
        x = torch.rand(1, 3, dtype=torch.double, requires_grad=True)
        test = gradcheck(init_grb, (x,), raise_exception=True)
        assert test
