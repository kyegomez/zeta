import torch
import pytest
from zeta.nn.modules.log_ff import LogFF, compute_entropy_safe


# Test fixture for a sample input tensor
@pytest.fixture
def sample_input():
    return torch.randn(32, 10)  # Adjust the batch size and input size as needed


# Test fixture for a sample LogFF model
@pytest.fixture
def sample_logff_model():
    return LogFF(10, 20, 30, 5)


# Test fixture for a sample LogFF model with usage tracking
@pytest.fixture
def sample_logff_model_with_usage():
    return LogFF(10, 20, 30, 5, usage_mode="soft")


# Test fixture for a sample LogFF model with dropout during training
@pytest.fixture
def sample_logff_model_with_dropout():
    return LogFF(10, 20, 30, 5, dropout=0.2)


# Test fixture for a sample LogFF model with region leakage during training
@pytest.fixture
def sample_logff_model_with_region_leak():
    return LogFF(10, 20, 30, 5, region_leak=0.1)


# Test fixture for a sample LogFF model with hardened decisions during training
@pytest.fixture
def sample_logff_model_with_hardened_decisions():
    return LogFF(10, 20, 30, 5, train_hardened=True)


# Test fixture for a sample LogFF model with entropy tracking
@pytest.fixture
def sample_logff_model_with_entropy():
    return LogFF(10, 20, 30, 5)


def test_logff_parameter_validation():
    with pytest.raises(ValueError):
        # Negative depth should raise an error
        LogFF(10, 20, 30, -5)
    with pytest.raises(ValueError):
        # Dropout > 1 should raise an error
        LogFF(10, 20, 30, 5, dropout=1.5)
    with pytest.raises(ValueError):
        # Region leak > 1 should raise an error
        LogFF(10, 20, 30, 5, region_leak=1.5)
    with pytest.raises(ValueError):
        # Invalid usage mode should raise an error
        LogFF(10, 20, 30, 5, usage_mode="invalid_mode")


def test_logff_forward(sample_logff_model, sample_input):
    output = sample_logff_model(sample_input)
    assert output.shape == (
        32,
        30,
    )  # Adjust expected shape based on your model parameters


def test_logff_forward_with_usage_tracking(sample_logff_model_with_usage, sample_input):
    output = sample_logff_model_with_usage(sample_input)
    assert output.shape == (
        32,
        30,
    )  # Adjust expected shape based on your model parameters


def test_logff_forward_with_dropout(sample_logff_model_with_dropout, sample_input):
    output = sample_logff_model_with_dropout(sample_input)
    assert output.shape == (
        32,
        30,
    )  # Adjust expected shape based on your model parameters


def test_logff_forward_with_region_leak(
    sample_logff_model_with_region_leak, sample_input
):
    output = sample_logff_model_with_region_leak(sample_input)
    assert output.shape == (
        32,
        30,
    )  # Adjust expected shape based on your model parameters


def test_logff_forward_with_hardened_decisions(
    sample_logff_model_with_hardened_decisions, sample_input
):
    output = sample_logff_model_with_hardened_decisions(sample_input)
    assert output.shape == (
        32,
        30,
    )  # Adjust expected shape based on your model parameters


def test_logff_forward_with_entropy(sample_logff_model_with_entropy, sample_input):
    output, entropies = sample_logff_model_with_entropy(
        sample_input, return_entropies=True
    )
    assert output.shape == (
        32,
        30,
    )  # Adjust expected shape based on your model parameters
    assert entropies.shape == (31,)  # Entropy shape should match the number of nodes
    # Ensure entropies are within a reasonable range
    assert (entropies >= 0).all()
    assert (entropies <= 0.6931).all()  # Maximum entropy for Bernoulli distribution
