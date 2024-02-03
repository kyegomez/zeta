import torch
from zeta.nn.modules._activations import (
    MishActivation,
    LinearActivation,
    LaplaceActivation,
    ReLUSquaredActivation,
)


# Tests for MishActivation
def test_mish_activation_initialization():
    activation = MishActivation()
    assert isinstance(activation, MishActivation)


def test_mish_activation_forward_positive():
    activation = MishActivation()
    x = torch.tensor([1.0, 2.0, 3.0])
    output = activation(x)
    # Expected values are approximations
    assert torch.allclose(
        output, torch.tensor([0.8651, 1.7924, 2.7306]), atol=1e-4
    )


def test_mish_activation_forward_negative():
    activation = MishActivation()
    x = torch.tensor([-1.0, -2.0, -3.0])
    output = activation(x)
    # Expected values are approximations
    assert torch.allclose(
        output, torch.tensor([-0.3034, -0.3297, -0.2953]), atol=1e-4
    )


# Tests for LinearActivation
def test_linear_activation_initialization():
    activation = LinearActivation()
    assert isinstance(activation, LinearActivation)


def test_linear_activation_forward():
    activation = LinearActivation()
    x = torch.tensor([1.0, 2.0, 3.0])
    output = activation(x)
    assert torch.equal(output, x)


# Tests for LaplaceActivation
def test_laplace_activation_initialization():
    activation = LaplaceActivation()
    assert isinstance(activation, LaplaceActivation)


def test_laplace_activation_forward():
    activation = LaplaceActivation()
    x = torch.tensor([1.0, 2.0, 3.0])
    output = activation(x)
    # Expected values are approximations
    assert torch.allclose(
        output, torch.tensor([0.6827, 0.8413, 0.9332]), atol=1e-4
    )


# Tests for ReLUSquaredActivation
def test_relusquared_activation_initialization():
    activation = ReLUSquaredActivation()
    assert isinstance(activation, ReLUSquaredActivation)


def test_relusquared_activation_forward_positive():
    activation = ReLUSquaredActivation()
    x = torch.tensor([1.0, 2.0, 3.0])
    output = activation(x)
    assert torch.allclose(output, torch.tensor([1.0, 4.0, 9.0]))


def test_relusquared_activation_forward_negative():
    activation = ReLUSquaredActivation()
    x = torch.tensor([-1.0, -2.0, -3.0])
    output = activation(x)
    assert torch.allclose(output, torch.tensor([0.0, 0.0, 0.0]))
