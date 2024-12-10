import numpy as np
import pytest
import torch

from zeta.utils import get_sinusoid_encoding_table


def test_basic_sinusoid_table():
    table = get_sinusoid_encoding_table(5, 4)
    assert table.shape == (1, 5, 4)


def test_zero_position_sinusoid_table():
    table = get_sinusoid_encoding_table(0, 4)
    assert table.size(1) == 0


def test_zero_dimension_sinusoid_table():
    table = get_sinusoid_encoding_table(5, 0)
    assert table.size(2) == 0


def test_negative_position_sinusoid_table():
    with pytest.raises(ValueError):
        get_sinusoid_encoding_table(-5, 4)


def test_negative_dimension_sinusoid_table():
    with pytest.raises(ValueError):
        get_sinusoid_encoding_table(5, -4)


@pytest.mark.parametrize("n_position, d_hid", [(10, 10), (5, 2), (100, 50)])
def test_sinusoid_table_parameters(n_position, d_hid):
    table = get_sinusoid_encoding_table(n_position, d_hid)
    assert table.shape == (1, n_position, d_hid)


def test_sinusoid_table_values():
    table = get_sinusoid_encoding_table(5, 4)
    base = np.array(
        [
            [pos / np.power(10000, 2 * (hid_j // 2) / 4) for hid_j in range(4)]
            for pos in range(5)
        ]
    )
    base[:, 0::2] = np.sin(base[:, 0::2])
    base[:, 1::2] = np.cos(base[:, 1::2])
    expected = torch.FloatTensor(base).unsqueeze(0)
    assert torch.allclose(
        table, expected, atol=1e-6
    )  # Allow for minor floating point differences


def test_sinusoid_table_return_type():
    table = get_sinusoid_encoding_table(5, 4)
    assert isinstance(table, torch.Tensor)
