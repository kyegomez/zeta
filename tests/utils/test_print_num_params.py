from unittest.mock import patch

import pytest
from torch import nn

from zeta.utils import print_num_params


@pytest.fixture
def simple_model():
    model = nn.Sequential(
        nn.Linear(2, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )
    return model


def test_num_params(simple_model):
    with patch("builtins.print") as mock_print:
        print_num_params(simple_model)
    mock_print.assert_called_once_with("Number of parameters in model: 16")


def test_num_params_zero():
    model = nn.Sequential()
    with patch("builtins.print") as mock_print:
        print_num_params(model)
    mock_print.assert_called_once_with("Number of parameters in model: 0")


def test_dist_available(simple_model):
    with patch("torch.distributed.is_available", return_value=True):
        with patch("torch.distributed.get_rank", return_value=0):
            with patch("builtins.print") as mock_print:
                print_num_params(simple_model)
    mock_print.assert_called_once_with("Number of parameters in model: 16")
