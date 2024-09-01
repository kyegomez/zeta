import pytest
import torch

from zeta.nn.modules.laser import Laser


def test_laser_init():
    laser = Laser(0.5)
    assert laser.rank_fraction == 0.5


def test_laser_forward_2d():
    laser = Laser(0.5)
    W = torch.randn(10, 10)
    W_approx = laser(W)
    assert W_approx.shape == W.shape


def test_laser_forward_3d():
    laser = Laser(0.5)
    W = torch.randn(5, 10, 10)
    W_approx = laser(W)
    assert W_approx.shape == W.shape


def test_laser_low_rank_approximation():
    laser = Laser(0.5)
    W = torch.randn(10, 10)
    W_approx = laser.low_rank_approximation(W)
    assert W_approx.shape == W.shape


def test_laser_rank_fraction_out_of_range():
    with pytest.raises(AssertionError):
        Laser(1.5)
