import torch
import torch.nn as nn
from zeta.nn.modules.adaptive_rmsnorm import AdaptiveRMSNorm


def test_adaptive_rmsnorm_init():
    arn = AdaptiveRMSNorm(10, dim_cond=5)
    assert isinstance(arn, AdaptiveRMSNorm)
    assert arn.dim_cond == 5
    assert arn.channel_first is False
    assert arn.scale == 10**0.5
    assert isinstance(arn.to_gamma, nn.Linear)
    assert arn.to_bias is None


def test_adaptive_rmsnorm_init_with_bias():
    arn = AdaptiveRMSNorm(10, dim_cond=5, bias=True)
    assert isinstance(arn.to_bias, nn.Linear)


def test_adaptive_rmsnorm_forward():
    arn = AdaptiveRMSNorm(10, dim_cond=5)
    x = torch.randn(2, 10)
    cond = torch.randn(2, 5)
    output = arn.forward(x, cond=cond)
    assert output.shape == (2, 10)


def test_adaptive_rmsnorm_forward_with_bias():
    arn = AdaptiveRMSNorm(10, dim_cond=5, bias=True)
    x = torch.randn(2, 10)
    cond = torch.randn(2, 5)
    output = arn.forward(x, cond=cond)
    assert output.shape == (2, 10)


def test_adaptive_rmsnorm_forward_channel_first():
    arn = AdaptiveRMSNorm(10, dim_cond=5, channel_first=True)
    x = torch.randn(2, 10, 3, 3)
    cond = torch.randn(2, 5)
    output = arn.forward(x, cond=cond)
    assert output.shape == (2, 10, 3, 3)
