import pytest
import torch
from zeta.ops.einops_poly import (
    rearrange_many,
    repeat_many,
    reduce_many,
    rearrange_with_anon_dims,
    repeat_with_anon_dims,
    reduce_with_anon_dims,
)

# Example input data
input_data = torch.randn(3, 4, 5, 6)


# Test rearrange_many function
@pytest.mark.parametrize("pattern", ["b h w c", "c b h w"])
def test_rearrange_many(pattern):
    output = list(rearrange_many([input_data, input_data], pattern=pattern))
    for tensor in output:
        assert tensor.shape == input_data.shape


# Test repeat_many function
@pytest.mark.parametrize("pattern", ["b h w c", "c b h w"])
def test_repeat_many(pattern):
    repeats = [2, 3]
    output = list(
        repeat_many([input_data, input_data], pattern=pattern, repeats=repeats)
    )
    for tensor in output:
        assert tensor.shape == (3 * repeats[0], 4 * repeats[1], 5, 6)


# Test reduce_many function
@pytest.mark.parametrize("pattern", ["b h w c", "c b h w"])
def test_reduce_many(pattern):
    output = list(
        reduce_many([input_data, input_data], pattern=pattern, reduction="mean")
    )
    for tensor in output:
        assert tensor.shape == (1, 1, 1, 1)


# Test rearrange_with_anon_dims function
@pytest.mark.parametrize("pattern", ["...a b c"])
@pytest.mark.parametrize("a_list", [(1, 2), (2, 3)])
def test_rearrange_with_anon_dims(pattern, a_list):
    output = rearrange_with_anon_dims(input_data, pattern=pattern, a=a_list)
    assert output.shape == (1, 2, 2, 3, 4, 5, 6)


# Test repeat_with_anon_dims function
@pytest.mark.parametrize("pattern", ["...a b c"])
@pytest.mark.parametrize("a_list", [(2, 3), (3, 4)])
def test_repeat_with_anon_dims(pattern, a_list):
    output = repeat_with_anon_dims(input_data, pattern=pattern, a=a_list)
    assert output.shape == (2, 3, 3, 4, 4, 5, 6)


# Test reduce_with_anon_dims function
@pytest.mark.parametrize("pattern", ["...a b c"])
@pytest.mark.parametrize("a_list", [(2, 3), (3, 4)])
def test_reduce_with_anon_dims(pattern, a_list):
    output = reduce_with_anon_dims(
        input_data, pattern=pattern, a=a_list, reduction="mean"
    )
    assert output.shape == (1, 1, 1, 2, 3, 4, 5, 6)


# Additional tests for rearrange_many function
def test_rearrange_many_invalid_pattern():
    with pytest.raises(ValueError):
        list(
            rearrange_many([input_data, input_data], pattern="invalid_pattern")
        )


def test_rearrange_many_with_multiple_patterns():
    patterns = ["b h w c", "c b h w", "h w b c"]
    output = list(rearrange_many([input_data, input_data], pattern=patterns))
    for tensor in output:
        assert tensor.shape == input_data.shape


# Additional tests for repeat_many function
def test_repeat_many_invalid_pattern():
    with pytest.raises(ValueError):
        list(
            repeat_many(
                [input_data, input_data],
                pattern="invalid_pattern",
                repeats=[2, 2],
            )
        )


def test_repeat_many_invalid_repeats():
    with pytest.raises(ValueError):
        list(
            repeat_many(
                [input_data, input_data], pattern="b h w c", repeats=[2]
            )
        )


def test_repeat_many_with_single_repeat():
    output = list(
        repeat_many([input_data, input_data], pattern="b h w c", repeats=[2, 1])
    )
    for tensor in output:
        assert tensor.shape == (6, 4, 5, 6)


# Additional tests for reduce_many function
def test_reduce_many_invalid_pattern():
    with pytest.raises(ValueError):
        list(
            reduce_many(
                [input_data, input_data],
                pattern="invalid_pattern",
                reduction="mean",
            )
        )


def test_reduce_many_invalid_reduction():
    with pytest.raises(ValueError):
        list(
            reduce_many(
                [input_data, input_data],
                pattern="b h w c",
                reduction="invalid_reduction",
            )
        )


def test_reduce_many_with_sum_reduction():
    output = list(
        reduce_many(
            [input_data, input_data], pattern="b h w c", reduction="sum"
        )
    )
    for tensor in output:
        assert tensor.shape == (1, 1, 1, 1)


# Additional tests for rearrange_with_anon_dims function
def test_rearrange_with_anon_dims_invalid_dim_list():
    with pytest.raises(ValueError):
        rearrange_with_anon_dims(input_data, pattern="...a b c", a=(1,))


def test_rearrange_with_anon_dims_invalid_pattern():
    with pytest.raises(ValueError):
        rearrange_with_anon_dims(
            input_data, pattern="invalid_pattern", a=[(1, 2), (2, 3)]
        )


# Additional tests for repeat_with_anon_dims function
def test_repeat_with_anon_dims_invalid_dim_list():
    with pytest.raises(ValueError):
        repeat_with_anon_dims(input_data, pattern="...a b c", a=(2,))


def test_repeat_with_anon_dims_invalid_pattern():
    with pytest.raises(ValueError):
        repeat_with_anon_dims(
            input_data, pattern="invalid_pattern", a=[(2, 3), (3, 4)]
        )


# Additional tests for reduce_with_anon_dims function
def test_reduce_with_anon_dims_invalid_dim_list():
    with pytest.raises(ValueError):
        reduce_with_anon_dims(input_data, pattern="...a b c", a=(2,))


def test_reduce_with_anon_dims_invalid_pattern():
    with pytest.raises(ValueError):
        reduce_with_anon_dims(
            input_data, pattern="invalid_pattern", a=[(2, 3), (3, 4)]
        )
