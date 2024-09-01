import torch

from zeta.utils import interpolate_pos_encoding_2d

# Note: You will need to import or define 'cast_if_src_dtype' function as it is used but not provided in the initial code snippet


def test_interpolate_same_target_size():
    r"""If the target_spatial_size is same as N, it should return the input pos_embed."""
    pos_embed = torch.rand((1, 36, 512))
    target_spatial_size = 36
    interpolated_pos_embed = interpolate_pos_encoding_2d(
        target_spatial_size, pos_embed
    )
    assert torch.equal(pos_embed, interpolated_pos_embed)


def test_interpolate_pos_encoding_2d_dimension():
    r"""The dimensions of the output tensor should be the same as input."""
    pos_embed = torch.rand((1, 36, 512))
    target_spatial_size = 72
    interpolated_pos_embed = interpolate_pos_encoding_2d(
        target_spatial_size, pos_embed
    )
    assert pos_embed.shape[:] == interpolated_pos_embed.shape[:]


def test_input_data_types():
    r"""The function should work correctly with different data types."""
    pos_embed = torch.rand((1, 36, 512), dtype=torch.float32)
    target_spatial_size = 72
    interpolated_pos_embed = interpolate_pos_encoding_2d(
        target_spatial_size, pos_embed
    )
    assert pos_embed.dtype == interpolated_pos_embed.dtype


def test_input_validation():
    r"""The function should raise an error if the inputs are invalid."""
    with pytest.raises(TypeError):
        interpolate_pos_encoding_2d("random_string", "random_string")
