import pytest
import torch
from zeta.utils import log


def test_log_zero():
    zero_tensor = torch.tensor(0.0)
    # checking if log function can handle inputs of zero
    assert log(zero_tensor) == torch.tensor(-46.0517)


def test_log_one():
    one_tensor = torch.tensor(1.0)
    # checking normal log behavior for positive numbers
    assert log(one_tensor) == torch.tensor(0.0)


def test_log_negative():
    negative_tensor = torch.tensor(-1.0)
    # testing log function with negative numbers
    with pytest.raises(ValueError):
        log(negative_tensor)


@pytest.mark.parametrize(
    "input_val, expected",
    [
        (torch.tensor(1e-20), torch.tensor(-46.0517)),
        (torch.tensor(2.0), torch.log(torch.tensor(2.0))),
    ],
)
def test_log_various_values(input_val, expected):
    # testing with a varied range of input values
    assert torch.isclose(log(input_val), expected, atol=1e-04)


def test_log_dtype():
    # Testing log with a tensor of type int
    tensor_int = torch.tensor(10)
    assert log(tensor_int).dtype == torch.float32
