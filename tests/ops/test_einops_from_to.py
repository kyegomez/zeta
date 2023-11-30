import pytest
import torch
from zeta.ops.einops_from_to import EinopsToAndFrom


# Fixture for creating a sample tensor
@pytest.fixture
def sample_tensor():
    return torch.randn(1, 2, 3, 4)


# Test the basic functionality of EinopsToAndFrom module
def test_einops_to_and_from_basic(sample_tensor):
    from_pattern = "b c h w"
    to_pattern = "b h w c"
    module = EinopsToAndFrom(from_pattern, to_pattern)
    output = module(sample_tensor)
    assert output.shape == (1, 3, 4, 2)


# Test with '...' in the from_pattern
def test_einops_to_and_from_with_anon_dims(sample_tensor):
    from_pattern = "...a c h w"
    to_pattern = "a h w c"
    module = EinopsToAndFrom(from_pattern, to_pattern)
    output = module(sample_tensor, a=[2])
    assert output.shape == (2, 3, 4, 1)


# Test with custom function that changes tensor values
def test_einops_to_and_from_with_custom_function(sample_tensor):
    from_pattern = "b c h w"
    to_pattern = "b h w c"

    def custom_fn(tensor, **kwargs):
        return tensor + 1

    module = EinopsToAndFrom(from_pattern, to_pattern)
    module.fn = custom_fn
    output = module(sample_tensor)
    assert torch.allclose(output, sample_tensor + 1)


# Test exception handling for invalid patterns
def test_einops_to_and_from_invalid_patterns(sample_tensor):
    from_pattern = "invalid_pattern"
    to_pattern = "b h w c"
    with pytest.raises(ValueError):
        module = EinopsToAndFrom(from_pattern, to_pattern)
        module(sample_tensor)


# Test exception handling for missing dimensions in reconstitution
def test_einops_to_and_from_missing_dimensions(sample_tensor):
    from_pattern = "b c h w"
    to_pattern = "b c w"
    module = EinopsToAndFrom(from_pattern, to_pattern)
    with pytest.raises(ValueError):
        module(sample_tensor)


# Test with multiple '...' in the from_pattern
def test_einops_to_and_from_multiple_anon_dims(sample_tensor):
    from_pattern = "...a ...b c h w"
    to_pattern = "a b h w c"
    module = EinopsToAndFrom(from_pattern, to_pattern)
    output = module(sample_tensor, a=[2], b=[3])
    assert output.shape == (2, 3, 4, 1)


# Test with custom function that changes tensor values with kwargs
def test_einops_to_and_from_custom_function_with_kwargs(sample_tensor):
    from_pattern = "b c h w"
    to_pattern = "b h w c"

    def custom_fn(tensor, **kwargs):
        a = kwargs["a"]
        return tensor + a

    module = EinopsToAndFrom(from_pattern, to_pattern)
    module.fn = custom_fn
    output = module(sample_tensor, a=5)
    assert torch.allclose(output, sample_tensor + 5)


# Test the module's backward pass with custom function
def test_einops_to_and_from_backward_pass(sample_tensor):
    from_pattern = "b c h w"
    to_pattern = "b h w c"

    def custom_fn(tensor, **kwargs):
        return tensor + 1

    module = EinopsToAndFrom(from_pattern, to_pattern)
    module.fn = custom_fn
    output = module(sample_tensor)

    # Perform backward pass
    loss = output.sum()
    loss.backward()

    # Ensure gradients are computed
    assert sample_tensor.grad is not None


# Test with non-default device (e.g., GPU)
def test_einops_to_and_from_device_placement():
    if torch.cuda.is_available():
        from_pattern = "b c h w"
        to_pattern = "b h w c"
        sample_tensor = torch.randn(1, 2, 3, 4).cuda()
        module = EinopsToAndFrom(from_pattern, to_pattern)
        module.to("cuda")
        output = module(sample_tensor)
        assert output.device == torch.device("cuda")
