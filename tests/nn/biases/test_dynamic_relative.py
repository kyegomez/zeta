import torch
from zeta.nn.biases.dynamic_position_bias import DynamicPositionBias


# Helper function to create random data
def create_random_data(shape):
    return torch.randn(shape)


# Helper function to check if two tensors are equal within a tolerance
def tensors_are_equal(tensor1, tensor2, tolerance=1e-6):
    return torch.allclose(tensor1, tensor2, atol=tolerance)


# Test case for initializing DynamicPositionBias
def test_dynamic_position_bias_init():
    dim = 512
    heads = 8
    bias = DynamicPositionBias(dim=dim, heads=heads)
    assert isinstance(bias, DynamicPositionBias)


# Test case for checking the forward pass of DynamicPositionBias
def test_dynamic_position_bias_forward():
    dim = 512
    heads = 8
    bias = DynamicPositionBias(dim=dim, heads=heads)

    i, j = 2, 4
    result = bias(i, j)

    # Check if the result has the correct shape
    assert result.shape == (heads, j - i, j - i)


# Test case for checking if the bias values are within the expected range
def test_dynamic_position_bias_values():
    dim = 512
    heads = 8
    bias = DynamicPositionBias(dim=dim, heads=heads)

    i, j = 2, 4
    result = bias(i, j)

    # Check if the bias values are within a reasonable range
    assert result.min() >= -1.0
    assert result.max() <= 1.0


# Test case for checking if the bias is on the correct device
def test_dynamic_position_bias_device():
    dim = 512
    heads = 8
    bias = DynamicPositionBias(dim=dim, heads=heads)

    assert bias.device == torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# Test case for checking if bias values are consistent for different instances of DynamicPositionBias
def test_dynamic_position_bias_values_consistency():
    dim = 512
    heads = 8
    i, j = 2, 4

    bias1 = DynamicPositionBias(dim=dim, heads=heads)
    bias2 = DynamicPositionBias(dim=dim, heads=heads)

    result1 = bias1(i, j)
    result2 = bias2(i, j)

    assert tensors_are_equal(result1, result2)


# Test case for checking if bias values are consistent for different positions
def test_dynamic_position_bias_position_consistency():
    dim = 512
    heads = 8
    i, j = 2, 4

    bias = DynamicPositionBias(dim=dim, heads=heads)

    result_i2_j4 = bias(i, j)
    result_i3_j5 = bias(i + 1, j + 1)

    assert tensors_are_equal(result_i2_j4, result_i3_j5)


# Test case for checking if bias values are consistent for different head counts
def test_dynamic_position_bias_head_count_consistency():
    dim = 512
    heads1 = 4
    heads2 = 8
    i, j = 2, 4

    bias1 = DynamicPositionBias(dim=dim, heads=heads1)
    bias2 = DynamicPositionBias(dim=dim, heads=heads2)

    result_heads4 = bias1(i, j)
    result_heads8 = bias2(i, j)

    assert tensors_are_equal(result_heads4, result_heads8)


# Test case for checking if device property is correctly set
def test_dynamic_position_bias_device_property():
    dim = 512
    heads = 8
    bias = DynamicPositionBias(dim=dim, heads=heads)

    expected_device = next(bias.parameters()).device
    assert bias.device == expected_device


# Test case for checking if bias values are within a reasonable range
def test_dynamic_position_bias_bias_values():
    dim = 512
    heads = 8
    bias = DynamicPositionBias(dim=dim, heads=heads)

    i, j = 2, 4
    result = bias(i, j)

    # Check if bias values are within a reasonable range
    assert torch.all(result >= -1.0)
    assert torch.all(result <= 1.0)


# Test case for checking if bias values match between different instances of DynamicPositionBias
def test_dynamic_position_bias_values_equal():
    dim = 512
    heads = 8
    i, j = 2, 4

    bias1 = DynamicPositionBias(dim=dim, heads=heads)
    bias2 = DynamicPositionBias(dim=dim, heads=heads)

    result1 = bias1(i, j)
    result2 = bias2(i, j)

    assert tensors_are_equal(result1, result2)
