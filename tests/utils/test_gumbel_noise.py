import pytest
import torch
from zeta.utils import gumbel_noise

# Basic Tests


def test_gumbel_noise():
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = gumbel_noise(tensor)
    assert isinstance(
        result, torch.Tensor
    ), "Output should be of type torch.Tensor"


# Test valid return values


def test_values():
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = gumbel_noise(tensor)
    # Since noise is a (0,1) uniform, gumbel noise should be in the range (-inf, +inf).
    # However, we don't expect to reach these limits in practice. Here we check that the
    # values are within a less extreme range.
    assert bool(
        ((result > -100) & (result < 100)).all()
    ), "Gumbel noise should fall within expected value range"


# Test invalid inputs


def test_tensor_requirement():
    with pytest.raises(TypeError):
        # gumbel_noise function expects a tensor as the input
        # but here a list is passed which should raise TypeError
        gumbel_noise([1.0, 2.0, 3.0])


# Parametrized Tests


@pytest.mark.parametrize(
    "input_tensor",
    [
        torch.tensor([1.0, 2.0, 3.0]),  # 1-D Tensor
        torch.tensor([[1, 2], [3, 4]]),  # 2-D Tensor
        torch.tensor(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        ),  # Higher Dimension Tensor
    ],
)
def test_gumbel_noise_dim(input_tensor):
    result = gumbel_noise(input_tensor)
    assert (
        result.shape == input_tensor.shape
    ), "Output tensor should have same dimensions as input"
