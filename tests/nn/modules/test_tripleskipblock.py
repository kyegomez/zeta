import pytest
import torch
import torch.nn as nn
from zeta.nn.modules import TripleSkipBlock


# Create Dummy Modules for Testing
class DummyModule(nn.Module):

    def forward(self, x):
        return x * 2


# A helper function to create an instance of TripleSkipBlock
@pytest.fixture
def triple_skip_block():
    module1 = module2 = module3 = DummyModule()
    return TripleSkipBlock(module1, module2, module3)


# Test for forward method
def test_forward(triple_skip_block):
    x = torch.tensor([1, 2, 3], dtype=torch.float32)
    output = triple_skip_block(x)
    assert torch.all(
        torch.eq(output, torch.tensor([15, 30, 45], dtype=torch.float32)))


# Test for correct instance creation
def test_instance_creation(triple_skip_block):
    assert isinstance(triple_skip_block.submodule1, DummyModule)
    assert isinstance(triple_skip_block.submodule2, DummyModule)
    assert isinstance(triple_skip_block.submodule3, DummyModule)


# Test for correct instance training mode
def test_training_mode(triple_skip_block):
    assert triple_skip_block.training is True
    triple_skip_block.eval()
    assert triple_skip_block.training is False


# Test to validate whether adding submodule modifies tensor correctly
@pytest.mark.parametrize(
    "input_tensor, expected_output",
    [
        (
            torch.tensor([1, 1, 1], dtype=torch.float32),
            torch.tensor([15, 15, 15], dtype=torch.float32),
        ),
        (
            torch.tensor([2, 2, 2], dtype=torch.float32),
            torch.tensor([30, 30, 30], dtype=torch.float32),
        ),
    ],
)
def test_with_different_inputs(triple_skip_block, input_tensor,
                               expected_output):
    output = triple_skip_block(input_tensor)
    assert torch.all(torch.eq(output, expected_output))
