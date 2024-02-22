import pytest
import torch

from zeta.utils import top_a

# logits map from [-1, 1] to [-inf, inf]
# top_a(logits, min_p_pow=2.0, min_p_ratio=0.02)
#   takes logits and returns a tensor of the same size
#   top_a does not return +inf, it caps at 1
#   top_a returns -inf if the input is -1


def test_top_a():
    logits = torch.Tensor([1.0, 0.0, -1.0])
    output = top_a(logits)
    assert torch.is_tensor(output), "Output should be a Torch tensor"
    assert (
        output.size() == logits.size()
    ), "Output size should match the input size"


@pytest.mark.parametrize(
    "logits, min_p_pow, min_p_ratio",
    [
        (torch.Tensor([1.0, 0.5, -0.2]), 2.0, 0.02),
        (torch.Tensor([-1.0, -0.5, -1.0]), 2.0, 0.02),
        (torch.Tensor([0.02, 0.001, -0.002]), 2.0, 0.02),
        (torch.Tensor([0.03, 0.0, -0.04]), 3.0, 0.02),
        (torch.Tensor([0.9999, -0.777, -0.0009]), 2.0, 0.10),
    ],
)
def test_top_a_values(logits, min_p_pow, min_p_ratio):
    output = top_a(logits, min_p_pow, min_p_ratio)
    assert torch.is_tensor(output), "Output should be a Torch tensor"
    assert (
        output.size() == logits.size()
    ), "Output size should match the input size"
    assert (output == float("-inf")).any() or (
        output == 1
    ).any(), (
        "Output elements should either be negative infinity or 1 (inclusive)"
    )


def test_top_a_exception():
    with pytest.raises(TypeError):
        top_a("non-tensor")


@pytest.fixture
def mock_tensor(monkeypatch):
    class MockTensor:
        def __init__(self):
            self.size_val = 3
            self.values = [1.0, 1.0, 1.0]

        def size(self):
            return self.size_val

    monkeypatch.setattr(torch, "Tensor", MockTensor)


def test_top_a_with_mock_tensor(mock_tensor):
    output = top_a(torch.Tensor())
    assert output.size() == mock_tensor.size()
    assert all(
        [val in output.values for val in mock_tensor.values]
    ), "Output values should match mocked tensor values"
