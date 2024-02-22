# LaplaceActivation

import math

import pytest
import torch

from zeta.nn import LaplaceActivation


def test_laplace_activation_forward_default_parameters():
    laplace_activation = LaplaceActivation()

    input = torch.tensor([0.5, 1.0, 2.0])
    output = laplace_activation.forward(input)

    expected_output = 0.5 * (
        1.0 + torch.erf((input - 0.707107) / (0.282095 * math.sqrt(2.0)))
    )

    assert torch.allclose(output, expected_output)


def test_laplace_activation_forward_custom_parameters():
    laplace_activation = LaplaceActivation()

    mu = 0.5
    sigma = 0.3
    input = torch.tensor([0.5, 1.0, 2.0])
    output = laplace_activation.forward(input, mu, sigma)

    expected_output = 0.5 * (
        1.0 + torch.erf((input - mu) / (sigma * math.sqrt(2.0)))
    )

    assert torch.allclose(output, expected_output)


def test_laplace_activation_forward_edge_case():
    # Edge case where input values are very large or very small
    laplace_activation = LaplaceActivation()

    input = torch.tensor([-1e6, 1e6])
    output = laplace_activation.forward(input)

    # Expected values would be 0.5 and 1.0 respectively.
    assert torch.allclose(output, torch.tensor([0.5, 1.0]))


@pytest.mark.parametrize(
    "input, mu, sigma, expected",
    [
        (
            torch.tensor([0.5, 1.0, 2.0]),
            0.5,
            0.3,
            torch.tensor([0.5, 0.5, 0.4795001]),
        ),
        (torch.tensor([-1e6, 1e6]), 0.5, 0.3, torch.tensor([0.0, 1.0])),
    ],
)
def test_laplace_activation_forward_params(input, mu, sigma, expected):
    laplace_activation = LaplaceActivation()

    output = laplace_activation.forward(input, mu, sigma)

    assert torch.allclose(output, expected)
