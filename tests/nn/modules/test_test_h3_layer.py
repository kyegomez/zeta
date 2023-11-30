from unittest.mock import Mock

import pytest
import torch

from zeta.nn.modules.h3 import H3Layer


# 1. Basic Tests
def test_h3_layer_creation():
    layer = H3Layer(256)
    assert isinstance(layer, H3Layer)


def test_forward_pass():
    layer = H3Layer(256)
    x = torch.randn(1, 256, 1024)
    output = layer(x)
    assert output.shape == torch.Size([1, 256, 1024])


# 2. Utilize Fixtures
@pytest.fixture
def sample_layer():
    return H3Layer(128)


def test_fixture_usage(sample_layer):
    x = torch.randn(1, 128, 1024)
    output = sample_layer(x)
    assert output.shape == torch.Size([1, 128, 1024])


# 3. Parameterized Testing
@pytest.mark.parametrize("dim", [128, 256, 512])
def test_parameterized_layer(dim):
    layer = H3Layer(dim)
    x = torch.randn(1, dim, 1024)
    output = layer(x)
    assert output.shape == torch.Size([1, dim, 1024])


def test_with_mocked_ssm():
    mock_ssm = Mock()
    layer = H3Layer(128)
    layer.diagonal_ssm = mock_ssm
    x = torch.randn(1, 128, 1024)
    layer(x)
    assert mock_ssm.called


# 5. Exception Testing
def test_invalid_dimension_raises_error():
    with pytest.raises(ValueError):
        H3Layer(0)


# 6. Test Coverage (requires pytest-cov)
def test_coverage():
    pytest.main(["--cov=your_module", "test_your_module.py"])


# Add more tests as needed...
