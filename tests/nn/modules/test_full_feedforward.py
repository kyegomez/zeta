import pytest
import torch

from zeta.nn.modules.feedforward import FeedForward


@pytest.fixture
def feed_forwardim():
    return FeedForward(768, 2048, 0.1)


def test_feed_forward_forward(feed_forwardim):
    x = torch.randn(1, 768)
    output = feed_forwardim(x)
    assert output.shape == (1, 2048)


def test_feed_forward_relu_squared(feed_forwardim):
    feed_forwardim_relu_squared = FeedForward(768, 2048, 0.1, relu_squared=True)
    x = torch.randn(1, 768)
    output = feed_forwardim_relu_squared(x)
    assert output.shape == (1, 2048)


def test_feed_forward_post_act_ln(feed_forwardim):
    feed_forwardim_post_act_ln = FeedForward(768, 2048, 0.1, post_act_ln=True)
    x = torch.randn(1, 768)
    output = feed_forwardim_post_act_ln(x)
    assert output.shape == (1, 2048)


def test_feed_forward_dropout(feed_forwardim):
    feed_forwardim_dropout = FeedForward(768, 2048, 0.5)
    x = torch.randn(1, 768)
    output = feed_forwardim_dropout(x)
    assert output.shape == (1, 2048)


def test_feed_forward_no_bias(feed_forwardim):
    feed_forwardim_no_bias = FeedForward(768, 2048, 0.1, no_bias=True)
    x = torch.randn(1, 768)
    output = feed_forwardim_no_bias(x)
    assert output.shape == (1, 2048)


def test_feed_forward_zero_init_output(feed_forwardim):
    feed_forwardim_zero_init_output = FeedForward(
        768, 2048, 0.1, zero_init_output=True
    )
    x = torch.randn(1, 768)
    output = feed_forwardim_zero_init_output(x)
    assert output.shape == (1, 2048)
    assert torch.allclose(output, torch.zeros_like(output))


def test_feed_forward_glu(feed_forwardim):
    feed_forwardim_glu = FeedForward(768, 2048, 0.1, glu=True)
    x = torch.randn(1, 768)
    output = feed_forwardim_glu(x)
    assert output.shape == (1, 2048)


def test_feed_forward_glu_mult_bias(feed_forwardim):
    feed_forwardim_glu_mult_bias = FeedForward(
        768, 2048, 0.1, glu=True, glu_mult_bias=True
    )
    x = torch.randn(1, 768)
    output = feed_forwardim_glu_mult_bias(x)
    assert output.shape == (1, 2048)


def test_feed_forward_swish(feed_forwardim):
    feed_forwardim_swish = FeedForward(768, 2048, 0.1, swish=True)
    x = torch.randn(1, 768)
    output = feed_forwardim_swish(x)
    assert output.shape == (1, 2048)


def test_feed_forward_input_dim_mismatch():
    with pytest.raises(ValueError):
        FeedForward(768, 1024, 0.1)(torch.randn(1, 512))


def test_feed_forward_negative_dropout():
    with pytest.raises(ValueError):
        FeedForward(768, 2048, -0.1)


def test_feed_forward_invalid_activation():
    with pytest.raises(ValueError):
        FeedForward(768, 2048, 0.1, activation="invalid")


def test_feed_forward_invalid_mult():
    with pytest.raises(ValueError):
        FeedForward(768, 2048, 1.5)


def test_feed_forward_invalid_dim_out():
    with pytest.raises(ValueError):
        FeedForward(768, dim_out=1024, dropout=0.1)


def test_feed_forward_invalid_glu_mult_bias():
    with pytest.raises(ValueError):
        FeedForward(768, 2048, 0.1, glu=True, glu_mult_bias=False)


def test_feed_forward_invalid_zero_init_output():
    with pytest.raises(ValueError):
        FeedForward(768, 2048, 0.1, zero_init_output=True, no_bias=True)


def test_feed_forward_invalid_no_bias():
    with pytest.raises(ValueError):
        FeedForward(768, 2048, 0.1, no_bias=True, glu=True)


def test_feed_forward_invalid_negative_dropout():
    with pytest.raises(ValueError):
        FeedForward(768, 2048, -0.1)


def test_feed_forward_invalid_swish_relu_squared():
    with pytest.raises(ValueError):
        FeedForward(768, 2048, 0.1, swish=True, relu_squared=True)


def test_feed_forward_invalid_swish_glu():
    with pytest.raises(ValueError):
        FeedForward(768, 2048, 0.1, swish=True, glu=True)


def test_feed_forward_invalid_relu_squared_glu():
    with pytest.raises(ValueError):
        FeedForward(768, 2048, 0.1, relu_squared=True, glu=True)


def test_feed_forward_invalid_relu_squared_post_act_ln():
    with pytest.raises(ValueError):
        FeedForward(768, 2048, 0.1, relu_squared=True, post_act_ln=True)


def test_feed_forward_dim_out_larger():
    feed_forwardim_dim_out_larger = FeedForward(768, 3072, 0.1)
    x = torch.randn(1, 768)
    output = feed_forwardim_dim_out_larger(x)
    assert output.shape == (1, 3072)


def test_feed_forward_dim_out_smaller():
    feed_forwardim_dim_out_smaller = FeedForward(768, 512, 0.1)
    x = torch.randn(1, 768)
    output = feed_forwardim_dim_out_smaller(x)
    assert output.shape == (1, 512)


# Add more edge cases and scenarios to cover other functionalities and edge cases.
