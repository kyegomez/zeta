# DualPathBlock

import pytest
import torch
import torch.nn as nn
from zeta.nn import DualPathBlock


class TestDualPathBlock:

    @pytest.fixture
    def simple_modules(self):
        return nn.Linear(10, 10), nn.Linear(10, 10)

    @pytest.fixture
    def mock_x(self):
        return torch.randn(1, 10)

    def test_initialization(self, simple_modules):
        block = DualPathBlock(*simple_modules)
        assert block.submodule1 == simple_modules[0]
        assert block.submodule2 == simple_modules[1]

    def test_forward(self, simple_modules, mock_x):
        block = DualPathBlock(*simple_modules)
        output = block(mock_x)
        assert isinstance(output, torch.Tensor)
        assert output.shape == mock_x.shape

    @pytest.mark.parametrize("input_shape, output_shape", [((1, 10), (1, 10)),
                                                           ((5, 10), (5, 10))])
    def test_shape_output(self, simple_modules, input_shape, output_shape):
        block = DualPathBlock(*simple_modules)
        mock_x = torch.randn(*input_shape)
        assert block(mock_x).shape == output_shape

    def test_submodule1_run(self, simple_modules, mock_x, mocker):
        submodule1_mock = mocker.Mock(side_effect=simple_modules[0])
        block = DualPathBlock(submodule1_mock, simple_modules[1])
        block(mock_x)
        submodule1_mock.assert_called_once_with(mock_x)

    def test_submodule2_run(self, simple_modules, mock_x, mocker):
        submodule2_mock = mocker.Mock(side_effect=simple_modules[1])
        block = DualPathBlock(simple_modules[0], submodule2_mock)
        block(mock_x)
        submodule2_mock.assert_called_once_with(mock_x)

    def test_forward_addition(self, simple_modules, mock_x):
        block = DualPathBlock(*simple_modules)
        expected_output = simple_modules[0](mock_x) + simple_modules[1](mock_x)
        assert torch.allclose(
            block(mock_x), expected_output, atol=1e-7
        )  # Use allclose because of potential floating point discrepancies
