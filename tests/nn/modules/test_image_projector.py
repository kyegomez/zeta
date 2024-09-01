import time

import pytest
import torch
import torch.nn as nn

from zeta.nn.modules.image_projector import ImagePatchCreatorProjector


# Create a fixture for a sample input tensor
@pytest.fixture
def sample_input_tensor():
    return torch.randn(1, 3, 64, 64)  # Shape: [B, C, H, W]


# Basic functionality test
def test_patch_projector_forward(sample_input_tensor):
    patch_projector = ImagePatchCreatorProjector(
        max_patch_size=16, embedding_dim=768
    )
    output_tensor = patch_projector(sample_input_tensor)
    assert output_tensor.shape == (
        1,
        256,
        768,
    )  # Check if the output shape matches expectations


# Exception testing
def test_patch_projector_exception_handling():
    patch_projector = ImagePatchCreatorProjector(
        max_patch_size=16, embedding_dim=768
    )
    # Test with invalid input tensor shape (negative dimension)
    invalid_input = torch.randn(1, -3, 64, 64)
    output_tensor = patch_projector(invalid_input)
    assert output_tensor is None  # Expecting None due to the exception


# Test dynamic patch size calculation
def test_patch_projector_dynamic_patch_size(sample_input_tensor):
    patch_projector = ImagePatchCreatorProjector(
        max_patch_size=16, embedding_dim=768
    )
    dynamic_patch_size = patch_projector.calculate_dynamic_patch_size(64, 64)
    assert dynamic_patch_size == 16  # Expecting the maximum patch size


# Test patch creation
def test_patch_projector_create_patches(sample_input_tensor):
    patch_projector = ImagePatchCreatorProjector(
        max_patch_size=16, embedding_dim=768
    )
    patch_size = 16
    patches = patch_projector.create_patches(sample_input_tensor, patch_size)
    assert patches.shape == (
        1,
        1024,
        16,
        16,
    )  # Expecting the correct shape of patches


# Test device placement
def test_patch_projector_device_placement(sample_input_tensor):
    if torch.cuda.is_available():
        patch_projector = ImagePatchCreatorProjector(
            max_patch_size=16, embedding_dim=768
        )
        sample_input_tensor = sample_input_tensor.cuda()
        patch_projector = patch_projector.cuda()
        output_tensor = patch_projector(sample_input_tensor)
        assert output_tensor.device == torch.device(
            "cuda"
        )  # Ensure output is on CUDA device


# Additional tests can be added to cover more cases, such as custom projection functions, edge cases, etc.


# Benchmarking test
def test_patch_projector_performance(sample_input_tensor):
    patch_projector = ImagePatchCreatorProjector(
        max_patch_size=16, embedding_dim=768
    )
    input_tensor = (
        sample_input_tensor.cuda()
        if torch.cuda.is_available()
        else sample_input_tensor
    )

    # Measure the time taken for 100 forward passes
    start_time = time.time()
    for _ in range(100):
        patch_projector(input_tensor)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time for 100 forward passes: {elapsed_time} seconds")

    # Assert that the forward passes are within a reasonable time frame
    assert elapsed_time < 1.0  # Adjust the threshold as needed


# Test case for device placement consistency
def test_patch_projector_device_placement_consistency(sample_input_tensor):
    patch_projector = ImagePatchCreatorProjector(
        max_patch_size=16, embedding_dim=768
    )
    sample_input_tensor = (
        sample_input_tensor.cuda()
        if torch.cuda.is_available()
        else sample_input_tensor
    )

    # Ensure consistent device placement
    output_tensor_1 = patch_projector(sample_input_tensor)
    output_tensor_2 = patch_projector(sample_input_tensor)
    assert output_tensor_1.device == output_tensor_2.device


# Test case for projection dimension consistency
def test_patch_projector_projection_dim_consistency(sample_input_tensor):
    patch_projector = ImagePatchCreatorProjector(
        max_patch_size=16, embedding_dim=768
    )
    input_tensor = (
        sample_input_tensor.cuda()
        if torch.cuda.is_available()
        else sample_input_tensor
    )

    output_tensor = patch_projector(input_tensor)
    assert (
        output_tensor.shape[-1] == 768
    )  # Ensure the output dimension is as expected


# Test case for patch size consistency
def test_patch_projector_patch_size_consistency(sample_input_tensor):
    patch_projector = ImagePatchCreatorProjector(
        max_patch_size=16, embedding_dim=768
    )
    input_tensor = (
        sample_input_tensor.cuda()
        if torch.cuda.is_available()
        else sample_input_tensor
    )

    dynamic_patch_size = patch_projector.calculate_dynamic_patch_size(64, 64)
    patches = patch_projector.create_patches(input_tensor, dynamic_patch_size)

    assert patches.shape[2] == patches.shape[3] == dynamic_patch_size


# Test case for invalid patch size
def test_patch_projector_invalid_patch_size():
    patch_projector = ImagePatchCreatorProjector(
        max_patch_size=16, embedding_dim=768
    )
    input_tensor = torch.randn(1, 3, 32, 32)  # Smaller image

    output_tensor = patch_projector(input_tensor)
    assert (
        output_tensor.shape[-1] == 768
    )  # Ensure the output dimension is as expected


# Test case for custom projection function
def test_patch_projector_custom_projection(sample_input_tensor):
    class CustomProjection(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.proj = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.proj(x)

    patch_projector = ImagePatchCreatorProjector(
        max_patch_size=16, embedding_dim=768
    )
    patch_projector.projection = CustomProjection(256, 768)
    input_tensor = (
        sample_input_tensor.cuda()
        if torch.cuda.is_available()
        else sample_input_tensor
    )

    output_tensor = patch_projector(input_tensor)
    assert (
        output_tensor.shape[-1] == 768
    )  # Ensure the output dimension is as expected


# Benchmarking test for different input sizes
@pytest.mark.parametrize(
    "input_shape", [(1, 3, 32, 32), (1, 3, 128, 128), (1, 3, 256, 256)]
)
def test_patch_projector_performance_various_input_sizes(
    sample_input_tensor, input_shape
):
    patch_projector = ImagePatchCreatorProjector(
        max_patch_size=16, embedding_dim=768
    )
    input_tensor = (
        sample_input_tensor.cuda()
        if torch.cuda.is_available()
        else sample_input_tensor
    )

    input_tensor = input_tensor.view(*input_shape)

    # Measure the time taken for 100 forward passes
    start_time = time.time()
    for _ in range(100):
        patch_projector(input_tensor)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(
        f"Elapsed time for 100 forward passes (Input Shape {input_shape}):"
        f" {elapsed_time} seconds"
    )

    # Assert that the forward passes are within a reasonable time frame
    assert (
        elapsed_time < 2.0
    )  # Adjust the threshold as needed for larger inputs


# Test case for output shape consistency
def test_patch_projector_output_shape_consistency(sample_input_tensor):
    patch_projector = ImagePatchCreatorProjector(
        max_patch_size=16, embedding_dim=768
    )
    input_tensor = (
        sample_input_tensor.cuda()
        if torch.cuda.is_available()
        else sample_input_tensor
    )

    dynamic_patch_size = patch_projector.calculate_dynamic_patch_size(64, 64)
    output_tensor = patch_projector(input_tensor)

    # Calculate the expected sequence length based on patch size and input dimensions
    expected_seq_len = (64 // dynamic_patch_size) * (64 // dynamic_patch_size)

    assert output_tensor.shape == (1, expected_seq_len, 768)


# Test case for edge case: invalid max_patch_size
def test_patch_projector_invalid_max_patch_size():
    with pytest.raises(ValueError):
        ImagePatchCreatorProjector(max_patch_size=0, embedding_dim=768)


# Test case for edge case: invalid embedding_dim
def test_patch_projector_invalid_embedding_dim():
    with pytest.raises(ValueError):
        ImagePatchCreatorProjector(max_patch_size=16, embedding_dim=0)


# Test case for edge case: invalid input tensor shape
def test_patch_projector_invalid_input_shape():
    patch_projector = ImagePatchCreatorProjector(
        max_patch_size=16, embedding_dim=768
    )
    input_tensor = torch.randn(1, 3, 32, 32)  # Smaller image

    with pytest.raises(ValueError):
        patch_projector(input_tensor)


# Test case for dynamic patch size calculation
def test_patch_projector_dynamic_patch_size_calculation():
    patch_projector = ImagePatchCreatorProjector(
        max_patch_size=16, embedding_dim=768
    )

    dynamic_patch_size = patch_projector.calculate_dynamic_patch_size(64, 128)
    assert dynamic_patch_size == 16


# Test case for changing max_patch_size and embedding_dim
def test_patch_projector_config_change(sample_input_tensor):
    patch_projector = ImagePatchCreatorProjector(
        max_patch_size=16, embedding_dim=768
    )
    input_tensor = (
        sample_input_tensor.cuda()
        if torch.cuda.is_available()
        else sample_input_tensor
    )

    output_tensor = patch_projector(input_tensor)

    # Change max_patch_size and embedding_dim
    patch_projector.max_patch_size = 32
    patch_projector.embedding_dim = 512

    new_output_tensor = patch_projector(input_tensor)

    # Ensure output tensors are different after configuration change
    assert not torch.allclose(output_tensor, new_output_tensor, atol=1e-7)


# Test case for random input tensor
def test_patch_projector_random_input():
    patch_projector = ImagePatchCreatorProjector(
        max_patch_size=16, embedding_dim=768
    )
    input_tensor = torch.randn(1, 3, 64, 64)  # Random input

    output_tensor = patch_projector(input_tensor)

    # Ensure the output tensor is not None
    assert output_tensor is not None
