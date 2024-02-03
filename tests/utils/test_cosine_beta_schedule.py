import torch
import pytest
from zeta.utils import cosine_beta_schedule


# Basic checks
def test_cosine_beta_schedule():
    assert cosine_beta_schedule(0).equal(torch.tensor([]))
    assert cosine_beta_schedule(1).equal(torch.tensor([0.9999]))


@pytest.mark.parametrize("timesteps", [10, 100, 1000])
def test_cosine_beta_schedule_length(timesteps):
    assert len(cosine_beta_schedule(timesteps)) == timesteps


def test_cosine_beta_schedule_values_range():
    """Ensure all values are in the range [0, 0.9999]"""
    for timesteps in range(100):
        betas = cosine_beta_schedule(timesteps)
        assert (betas >= 0).all() and (betas <= 0.9999).all()


def test_cosine_beta_schedule_values_decreasing():
    for timesteps in range(100):
        betas = cosine_beta_schedule(timesteps)
        assert (betas[:-1] >= betas[1:]).all()


# Test with negative timesteps values
def test_cosine_beta_schedule_negative_timesteps():
    with pytest.raises(RuntimeError):
        cosine_beta_schedule(-10)


# Test with floating timesteps values
def test_cosine_beta_schedule_float_timesteps():
    with pytest.raises(TypeError):
        cosine_beta_schedule(10.5)


# Test large values
@pytest.mark.slow
def test_cosine_beta_schedule_large_timesteps():
    assert len(cosine_beta_schedule(1e6)) == 1e6


# Test using mathematical calculation
def test_cosine_beta_schedule_math():
    for timesteps in range(1, 100):
        betas = cosine_beta_schedule(timesteps)
        x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        expected_betas = 1 - (
            torch.cos(
                ((x[1:] / timesteps) + 0.008) / (1 + 0.008) * torch.pi * 0.5
            )
            ** 2
            / torch.cos(
                ((x[:-1] / timesteps) + 0.008) / (1 + 0.008) * torch.pi * 0.5
            )
            ** 2
        )
        expected_betas = torch.clip(expected_betas, 0, 0.9999)
        assert torch.allclose(betas, expected_betas, atol=1e-7)
