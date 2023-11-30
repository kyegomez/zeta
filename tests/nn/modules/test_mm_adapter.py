import pytest
import torch
from zeta.nn.modules.mm_adapter import MultiModalAdapterDenseNetwork


# Define a fixture for creating an instance of the MultiModalAdapterDenseNetwork
@pytest.fixture
def mm_adapter():
    return MultiModalAdapterDenseNetwork(dim=512, hidden_dim=1024, depth=3)


# Example of a basic test
def test_creation(mm_adapter):
    assert isinstance(mm_adapter, MultiModalAdapterDenseNetwork)


# Example of a parameterized test with different input dimensions
@pytest.mark.parametrize("dim", [256, 512, 1024])
def test_input_dimensions(dim):
    mm_adapter = MultiModalAdapterDenseNetwork(dim=dim)
    assert mm_adapter.dim == dim


# Example of a test for the forward pass
def test_forward_pass(mm_adapter):
    input_tensor = torch.randn(1, mm_adapter.dim)
    output_tensor = mm_adapter(input_tensor)
    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.shape == (1, mm_adapter.dim)


# Example of a test for layer normalization
def test_layer_normalization(mm_adapter):
    input_tensor = torch.randn(1, mm_adapter.dim)
    normalized_tensor = mm_adapter.norm(input_tensor)
    assert isinstance(normalized_tensor, torch.Tensor)
    assert normalized_tensor.shape == (1, mm_adapter.dim)


# Example of a test for skip connections
def test_skip_connections(mm_adapter):
    input_tensor = torch.randn(1, mm_adapter.dim)
    output_tensor = mm_adapter(input_tensor)
    assert torch.allclose(input_tensor + input_tensor, output_tensor)


# Example of a test for activation function
def test_activation_function(mm_adapter):
    input_tensor = torch.randn(1, mm_adapter.dim)
    output_tensor = mm_adapter(input_tensor)
    assert torch.allclose(torch.nn.SiLU()(input_tensor), output_tensor)


# Example of a test for the depth of the network
def test_depth(mm_adapter):
    assert mm_adapter.depth == 3


def test_proj_layer(mm_adapter):
    input_tensor = torch.randn(1, mm_adapter.dim)
    projected_tensor = mm_adapter.proj(input_tensor)
    assert isinstance(projected_tensor, torch.Tensor)
    assert projected_tensor.shape == (1, mm_adapter.dim)


def test_silu_activation(mm_adapter):
    input_tensor = torch.randn(1, mm_adapter.dim)
    activated_tensor = mm_adapter.silu(input_tensor)
    assert isinstance(activated_tensor, torch.Tensor)
    assert activated_tensor.shape == (1, mm_adapter.dim)


def test_skip_connection(mm_adapter):
    input_tensor1 = torch.randn(1, mm_adapter.dim)
    input_tensor2 = torch.randn(1, mm_adapter.dim)
    output_tensor = mm_adapter.skip_connections(input_tensor1, input_tensor2)
    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.shape == (1, mm_adapter.dim)


# Add more tests covering different aspects of the class...

# You can continue adding more tests as needed...
