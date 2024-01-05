import torch
import pytest
from zeta.nn.modules.laser import LASER


def test_laser_init():
    laser = LASER(0.5)
    assert laser.rank_fraction == 0.5


def test_laser_forward_2d():
    laser = LASER(0.5)
    W = torch.randn(10, 10)
    W_approx = laser(W)
    assert W_approx.shape == W.shape


def test_laser_forward_3d():
    laser = LASER(0.5)
    W = torch.randn(5, 10, 10)
    W_approx = laser(W)
    assert W_approx.shape == W.shape


def test_laser_low_rank_approximation():
    laser = LASER(0.5)
    W = torch.randn(10, 10)
    W_approx = laser.low_rank_approximation(W)
    assert W_approx.shape == W.shape


def test_laser_rank_fraction_out_of_range():
    with pytest.raises(AssertionError):
        LASER(1.5)
