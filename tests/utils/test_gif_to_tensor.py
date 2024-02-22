import PIL
import pytest
import torch
from PIL import Image

from zeta.utils import gif_to_tensor


# Mock of the seek_all_images function to simulate various outputs
def mock_seek_all_images(img, channels):
    return [img] * channels


# Fixture for a mock GIF image to be used in tests
@pytest.fixture
def mock_image(monkeypatch):
    monkeypatch.setattr("zeta.utils.seek_all_images", mock_seek_all_images)
    return Image.new("RGB", (60, 30))


# Basic test case for successful function operation
def test_gif_to_tensor_basic(mock_image):
    result = gif_to_tensor(mock_image, channels=3)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 3, 60, 30)


# Tests for various number of channels
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_gif_to_tensor_channels(mock_image, channels):
    result = gif_to_tensor(mock_image, channels=channels)
    assert result.shape == (channels, channels, 60, 30)


# Test for non-existent file path, expecting a FileNotFound error
def test_gif_to_tensor_invalid_path():
    with pytest.raises(FileNotFoundError):
        gif_to_tensor("non_existent.gif")


# Test for file that is not of an image type, expecting an UnidentifiedImageError
def test_gif_to_tensor_non_image_file():
    with pytest.raises(PIL.UnidentifiedImageError):
        gif_to_tensor("some_file.txt")


# TODO: Add more tests based on the function's specification like invalid image format, invalid transform function etc.
