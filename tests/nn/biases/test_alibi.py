from einops import rearrange
import torch
from torch import nn
from zeta.nn.biases.alibi import (
    AlibiPositionalBias,
    LearnedAlibiPositionalBias,
    pad_at_dim,
)
from zeta.utils.main import exists


# Helper function to create a bias tensor
def create_bias_tensor(i, j, num_heads):
    bias = torch.zeros(num_heads, 1, i, j)
    return bias


# Helper function to create a slope tensor
def create_slope_tensor(num_heads):
    slopes = torch.tensor(AlibiPositionalBias._get_slopes(num_heads))
    return slopes.view(num_heads, 1, 1)


# Helper function to create a learned log slopes tensor
def create_learned_logslopes_tensor(num_heads):
    logslopes = torch.log(
        torch.tensor(AlibiPositionalBias._get_slopes(num_heads)))
    return nn.Parameter(logslopes)


# Test case for creating an instance of AlibiPositionalBias
def test_alibi_positional_bias_init():
    bias = AlibiPositionalBias(heads=8, num_heads=4)
    assert isinstance(bias, AlibiPositionalBias)


# Test case for creating an instance of LearnedAlibiPositionalBias
def test_learned_alibi_positional_bias_init():
    bias = LearnedAlibiPositionalBias(heads=8, num_heads=4)
    assert isinstance(bias, LearnedAlibiPositionalBias)


# Test case for computing bias using AlibiPositionalBias
def test_alibi_positional_bias_forward():
    num_heads = 4
    i, j = 2, 3
    bias = AlibiPositionalBias(heads=8, num_heads=num_heads)
    result = bias(i, j)
    assert result.shape == (num_heads, 1, i, j)


# Test case for computing bias using LearnedAlibiPositionalBias
def test_learned_alibi_positional_bias_forward():
    num_heads = 4
    i, j = 2, 3
    bias = LearnedAlibiPositionalBias(heads=8, num_heads=num_heads)
    result = bias(i, j)
    assert result.shape == (num_heads, 1, i, j)


# Test case for padding a tensor at a specified dimension
def test_pad_at_dim():
    tensor = torch.ones(2, 2)
    pad = (2, 3)
    result = pad_at_dim(tensor, pad, dim=-1)
    assert result.shape == (2, 5)


# Test case for creating a bias tensor
def test_create_bias_tensor():
    i, j, num_heads = 2, 3, 4
    bias = create_bias_tensor(i, j, num_heads)
    assert bias.shape == (num_heads, 1, i, j)


# Test case for creating a slope tensor
def test_create_slope_tensor():
    num_heads = 4
    slopes = create_slope_tensor(num_heads)
    assert slopes.shape == (num_heads, 1, 1)


# Test case for creating a learned log slopes tensor
def test_create_learned_logslopes_tensor():
    num_heads = 4
    logslopes = create_learned_logslopes_tensor(num_heads)
    assert logslopes.shape == (num_heads,)


# Test case for getting the device of a tensor
def test_device_property():
    num_heads = 4
    bias = AlibiPositionalBias(heads=8, num_heads=num_heads)
    device = bias.device
    assert isinstance(device, torch.device)


# Test case for computing bias with AlibiPositionalBias with existing bias
def test_alibi_positional_bias_existing_bias():
    num_heads = 4
    i, j = 2, 3
    bias = AlibiPositionalBias(heads=8, num_heads=num_heads)
    bias(i, j)  # Create bias tensor
    result = bias(i, j)
    assert result.shape == (num_heads, 1, i, j)


# Test case for computing bias with LearnedAlibiPositionalBias with existing bias
def test_learned_alibi_positional_bias_existing_bias():
    num_heads = 4
    i, j = 2, 3
    bias = LearnedAlibiPositionalBias(heads=8, num_heads=num_heads)
    bias(i, j)  # Create bias tensor
    result = bias(i, j)
    assert result.shape == (num_heads, 1, i, j)


# Test case for gradient checking of AlibiPositionalBias
def test_alibi_positional_bias_gradient_check():
    num_heads = 4
    i, j = 2, 3
    bias = AlibiPositionalBias(heads=8, num_heads=num_heads)
    i_tensor = torch.tensor(i, dtype=torch.float32, requires_grad=True)
    j_tensor = torch.tensor(j, dtype=torch.float32, requires_grad=True)
    result = bias(i_tensor, j_tensor)
    grad_output = torch.randn_like(result)
    torch.autograd.gradcheck(bias, (i_tensor, j_tensor), grad_output)


# Test case for gradient checking of LearnedAlibiPositionalBias
def test_learned_alibi_positional_bias_gradient_check():
    num_heads = 4
    i, j = 2, 3
    bias = LearnedAlibiPositionalBias(heads=8, num_heads=num_heads)
    i_tensor = torch.tensor(i, dtype=torch.float32, requires_grad=True)
    j_tensor = torch.tensor(j, dtype=torch.float32, requires_grad=True)
    result = bias(i_tensor, j_tensor)
    grad_output = torch.randn_like(result)
    torch.autograd.gradcheck(bias, (i_tensor, j_tensor), grad_output)


# Helper function to create a sample tensor
def create_sample_tensor(shape):
    return torch.randn(*shape)


# Helper function to check if two tensors are equal
def tensors_equal(tensor1, tensor2):
    return torch.allclose(tensor1, tensor2, atol=1e-6)


# Test for the existence of a helper function exists
def test_exists_function():
    assert exists(None) is False
    assert exists(0) is True
    assert exists("Hello") is True


# Test for the pad_at_dim helper function
def test_pad_at_dim_function():
    tensor = torch.tensor([1, 2, 3])
    padded_tensor = pad_at_dim(tensor, (2, 2), dim=-1, value=0)
    assert tensors_equal(padded_tensor, torch.tensor([0, 0, 1, 2, 3, 0, 0]))


# Test for the tensors_equal helper function
def test_tensors_equal_function():
    tensor1 = torch.tensor([1.0, 2.0, 3.0])
    tensor2 = torch.tensor([1.0, 2.0, 3.0])
    tensor3 = torch.tensor([1.0, 2.0, 3.1])

    assert tensors_equal(tensor1, tensor2) is True
    assert tensors_equal(tensor1, tensor3) is False


# Additional tests for tensor manipulation functions


# Test for the create_sample_tensor function
def test_create_sample_tensor_function():
    shape = (2, 3, 4)
    tensor = create_sample_tensor(shape)
    assert tensor.shape == shape


# Test for rearrange function from einops
def test_einops_rearrange_function():
    tensor = torch.randn(2, 3, 4)
    rearranged_tensor = rearrange(tensor, "a b c -> b a c")
    assert rearranged_tensor.shape == (3, 2, 4)


# Test for the nn.Module class inheritance
def test_nn_module_inheritance():
    assert issubclass(AlibiPositionalBias, nn.Module) is True
    assert issubclass(LearnedAlibiPositionalBias, nn.Module) is True


# Helper function to create random data
def create_random_data(shape):
    return torch.randn(shape)


# Helper function to check if two tensors are equal within a tolerance
def tensors_are_equal(tensor1, tensor2, tolerance=1e-6):
    return torch.allclose(tensor1, tensor2, atol=tolerance)


# Test case for checking if slopes are computed correctly in AlibiPositionalBias
def test_alibi_positional_bias_slopes():
    num_heads = 8
    bias = AlibiPositionalBias(heads=num_heads, num_heads=num_heads)

    expected_slopes = torch.tensor(bias._get_slopes(num_heads))
    assert tensors_are_equal(bias.slopes, expected_slopes)


# Test case for checking if slopes are learned correctly in LearnedAlibiPositionalBias
def test_learned_alibi_positional_bias_slopes():
    num_heads = 8
    bias = LearnedAlibiPositionalBias(heads=num_heads, num_heads=num_heads)

    expected_slopes = torch.tensor(bias._get_slopes(num_heads))
    expected_slopes_exp = torch.exp(expected_slopes)

    assert tensors_are_equal(bias.learned_logslopes.exp(), expected_slopes_exp)


# Test case for checking if bias values match between AlibiPositionalBias and LearnedAlibiPositionalBias
def test_alibi_vs_learned_bias_values():
    num_heads = 4
    i, j = 2, 4

    alibi_bias = AlibiPositionalBias(heads=num_heads, num_heads=num_heads)
    learned_bias = LearnedAlibiPositionalBias(heads=num_heads,
                                              num_heads=num_heads)

    alibi_result = alibi_bias(i, j)
    learned_result = learned_bias(i, j)

    assert tensors_are_equal(alibi_result, learned_result)


# Test case for checking if bias values match between different instances of AlibiPositionalBias
def test_alibi_bias_values_equal():
    num_heads = 4
    i, j = 2, 4

    bias1 = AlibiPositionalBias(heads=num_heads, num_heads=num_heads)
    bias2 = AlibiPositionalBias(heads=num_heads, num_heads=num_heads)

    result1 = bias1(i, j)
    result2 = bias2(i, j)

    assert tensors_are_equal(result1, result2)


# Test case for checking if bias values match between different instances of LearnedAlibiPositionalBias
def test_learned_bias_values_equal():
    num_heads = 4
    i, j = 2, 4

    bias1 = LearnedAlibiPositionalBias(heads=num_heads, num_heads=num_heads)
    bias2 = LearnedAlibiPositionalBias(heads=num_heads, num_heads=num_heads)

    result1 = bias1(i, j)
    result2 = bias2(i, j)

    assert tensors_are_equal(result1, result2)
