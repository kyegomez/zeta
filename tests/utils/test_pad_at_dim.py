import pytest
import torch

from zeta.utils import pad_at_dim


def test_pad_at_dim():
    tensor = torch.tensor([1, 2, 3, 4])
    pad = (1, 1)
    padded_tensor = pad_at_dim(tensor, pad)
    assert padded_tensor.tolist() == [0, 1, 2, 3, 4, 0]


def test_pad_at_last_dim():
    tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    pad = (1, 1)
    padded_tensor = pad_at_dim(tensor, pad)
    assert padded_tensor.tolist() == [[0, 1, 2, 3, 4, 0], [0, 5, 6, 7, 8, 0]]


def test_pad_at_first_dim():
    tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    pad = (1, 1)
    padded_tensor = pad_at_dim(tensor, pad, 0)
    assert padded_tensor.tolist() == [
        [0, 0, 0, 0, 0],
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [0, 0, 0, 0, 0],
    ]


def test_pad_at_negative_dim():
    tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    pad = (1, 1)
    padded_tensor = pad_at_dim(tensor, pad, -1)
    assert padded_tensor.tolist() == [[0, 1, 2, 3, 4, 0], [0, 5, 6, 7, 8, 0]]


def test_pad_with_value():
    tensor = torch.tensor([1, 2, 3, 4])
    pad = (1, 1)
    padded_tensor = pad_at_dim(tensor, pad, value=9)
    assert padded_tensor.tolist() == [9, 1, 2, 3, 4, 9]


@pytest.mark.parametrize("pad", [(1, 1), (2, 2), (3, 3), (4, 4)])
def test_different_pad_sizes(pad):
    tensor = torch.tensor([1, 2, 3, 4])
    padded_tensor = pad_at_dim(tensor, pad)
    assert padded_tensor[0] == 0
    assert padded_tensor[-1] == 0


@pytest.mark.parametrize("dim", [-1, 0, 1, 2, 3])
def test_pad_at_different_dims(dim):
    tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    pad_at_dim(tensor, (1, 1), dim)
    # Add corresponding asserts based on value of dim
