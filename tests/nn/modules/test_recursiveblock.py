# RecursiveBlock

import pytest
import torch
import torch.nn as nn
from zeta.nn import RecursiveBlock


def test_recursive_block_initialization():
    block = RecursiveBlock(nn.Linear(10, 10), 5)
    assert isinstance(block.modules, nn.Module)
    assert isinstance(block.iters, int)


def test_recursive_block_forward_pass():
    module = nn.Linear(10, 10)
    block = RecursiveBlock(module, 2)
    input_tensor = torch.randn(3, 10)
    output_tensor = block(input_tensor)
    assert output_tensor.shape == torch.Size([3, 10])


def test_recursive_block_fail_with_zero_iterations():
    with pytest.raises(ValueError):
        RecursiveBlock(2, nn.Linear(10, 10))


def test_recursive_block_fail_with_negative_iterations():
    with pytest.raises(ValueError):
        RecursiveBlock(-1, nn.Linear(10, 10))


@pytest.mark.parametrize("num_iterations", [1, 2, 3, 4, 5])
def test_recursive_block_iteration_count(num_iterations):
    input_tensor = torch.ones(1, 10)
    module = nn.Linear(10, 10)
    module.weight.data.fill_(1)
    module.bias.data.fill_(1)
    block = RecursiveBlock(module, num_iterations)
    output_tensor = block(input_tensor)
    # The output tensor should equal the input_tensor after applying the module "num_iterations" times
    assert torch.all(output_tensor == torch.ones(1, 10) * num_iterations + 1)


def test_recursive_block_not_a_module():
    with pytest.raises(TypeError):
        RecursiveBlock("not_a_module", 2)


def test_recursive_block_wrong_positional_arguments():
    with pytest.raises(TypeError):
        RecursiveBlock(2, "not_a_module")


def test_recursive_block_extra_kwargs():
    with pytest.raises(TypeError):
        RecursiveBlock(2, nn.Linear(10, 10), extra_kwarg=False)


# ... Create more tests with different nn.Modules (not just nn.Linear), different edge cases, etc.
