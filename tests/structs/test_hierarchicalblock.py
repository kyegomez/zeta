import pytest
import torch
from zeta.nn import HierarchicalBlock


def test_HierarchicalBlock_init():
    hb = HierarchicalBlock(64)
    assert hb.stride == 1
    assert hb.compress_factor == 1
    assert hb.no_compress is True
    assert hb.has_attn is False
    assert hb.attn is None


def test_HierarchicalBlock_forward():
    hb = HierarchicalBlock(64)
    x = torch.randn((1, 64, 64))
    result = hb.forward(x)
    assert result.shape == x.shape


def test_HierarchicalBlock_raises():
    with pytest.raises(AssertionError):
        # compression factor is not a power of 2
        HierarchicalBlock(64, compress_factor=3)

    with pytest.raises(AssertionError):
        # window size is negative
        HierarchicalBlock(64, window_size=-5)


@pytest.mark.parametrize(
    "dim, dim_head, heads, window_size, compress_factor, stride, ff_mult",
    [
        # some examples
        (64, 32, 4, 5, 2, 1, 1),
        (32, 16, 2, 3, 4, 2, 2),
        # edge cases
        (0, 0, 0, 0, 1, 0, 0),
    ],
)
def test_HierarchicalBlock_dim(
    dim, dim_head, heads, window_size, compress_factor, stride, ff_mult
):
    # Test if correct exceptions are raised when dimensions are zero or negative
    try:
        HierarchicalBlock(
            dim,
            dim_head,
            heads,
            window_size,
            compress_factor,
            stride,
        )
    except ValueError:
        assert (
            dim <= 0
            or dim_head <= 0
            or heads <= 0
            or window_size < 0
            or compress_factor <= 0
            or stride <= 0
            or ff_mult <= 0
        )
