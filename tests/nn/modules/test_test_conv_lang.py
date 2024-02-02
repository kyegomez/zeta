from unittest.mock import Mock

import pytest
import torch
from torch import nn

from zeta.nn.modules.lang_conv_module import ConvolutionLanguageBlock


# 1. Basic Tests
def test_convolution_language_block_creation():
    block = ConvolutionLanguageBlock(256, 512, 3, 1)
    assert isinstance(block, ConvolutionLanguageBlock)


def test_forward_pass():
    block = ConvolutionLanguageBlock(256, 512, 3, 1)
    x = torch.randn(1, 256, 1024)
    output = block(x)
    assert output.shape == torch.Size([1, 512, 1024])


# 2. Utilize Fixtures
@pytest.fixture
def sample_block():
    return ConvolutionLanguageBlock(128, 256, 3, 1)


def test_fixture_usage(sample_block):
    x = torch.randn(1, 128, 1024)
    output = sample_block(x)
    assert output.shape == torch.Size([1, 256, 1024])


# 3. Parameterized Testing
@pytest.mark.parametrize(
    ("in_channels, out_channels, kernel_size, padding, depth, stride,"
     " activation, batchnorm, dilation, dropout"),
    [
        (128, 256, 3, 1, 2, 1, "relu", True, 1, 0.1),
        (256, 512, 3, 1, 3, 1, "gelu", False, 2, 0.2),
        # Add more parameter combinations as needed
    ],
)
def test_parameterized_block(
    in_channels,
    out_channels,
    kernel_size,
    padding,
    depth,
    stride,
    activation,
    batchnorm,
    dilation,
    dropout,
):
    block = ConvolutionLanguageBlock(
        in_channels,
        out_channels,
        kernel_size,
        padding,
        depth,
        stride,
        activation,
        batchnorm,
        dilation,
        dropout,
    )
    x = torch.randn(1, in_channels, 1024)
    output = block(x)
    assert output.shape == torch.Size([1, out_channels, 1024])


def test_with_mocked_convolution_layer():
    mock_convolution = Mock(spec=nn.Conv1d)
    block = ConvolutionLanguageBlock(128, 256, 3, 1)
    block.conv_layers[0] = mock_convolution
    x = torch.randn(1, 128, 1024)
    block(x)
    assert mock_convolution.called


# 5. Exception Testing
def test_invalid_activation_raises_error():
    with pytest.raises(ValueError):
        ConvolutionLanguageBlock(128,
                                 256,
                                 3,
                                 1,
                                 activation="invalid_activation")
