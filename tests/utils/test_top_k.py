import pytest
import torch
from math import ceil
from zeta.utils import top_k


def test_top_k_positive_case():
    logits = torch.randn(1, 10)
    probs = top_k(logits, 0.9)
    k = ceil((1 - 0.9) * logits.shape[-1])
    assert probs.shape == logits.shape
    assert (probs[probs != float("-inf")].numel() == k
           )  # checks number of elements that aren't negative infinity


def test_dimensions_positive_case():
    logits = torch.randn(
        1, 5, 5)  # assumed example for logits with more than 2 dimensions
    top_k(logits, 0.9)


@pytest.mark.parametrize(
    "threshold",
    [
        (0.8),
        (0.9),
        (1),
    ],
)
def test_top_k_threshold_variations(threshold):
    logits = torch.randn(1, 5)
    probs = top_k(logits, threshold)
    k = ceil((1 - threshold) * logits.shape[-1])
    assert probs[probs != float("-inf")].numel() == k


def test_top_k_large_values():
    logits = torch.randn(1, 1000)
    threshold = 0.9
    probs = top_k(logits, threshold)
    k = ceil((1 - threshold) * logits.shape[-1])
    assert probs[probs != float("-inf")].numel() == k


def test_top_k_empty_input():
    with pytest.raises(
            Exception
    ):  # assuming that you would want to handle this case with an exception
        top_k(torch.tensor([]), 0.8)
