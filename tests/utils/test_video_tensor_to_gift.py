import pytest
import torch
from unittest.mock import MagicMock, patch
from PIL import Image
from zeta.utils import video_tensor_to_gift


def setup_test_tensor():
    test_tensor = torch.rand((5, 5, 3))
    return test_tensor


def setup_test_pil_image():
    return Image.new("RGB", (5, 5))


@pytest.fixture
def tensor(tmpdir):
    tensor = setup_test_tensor()
    return tensor


@pytest.fixture
def test_image():
    img = setup_test_pil_image()
    return img


@pytest.mark.parametrize(
    "duration, loop, optimize",
    [
        (120, 0, True),
        (60, 1, False),
        (240, 2, True),
        (0, 0, False),
        (180, 1, True),
    ],
)
def test_video_tensor_to_gif_valid_params(duration, loop, optimize, tensor,
                                          test_image):
    path = "/test/path"

    with patch("torchvision.transforms.ToPILImage") as mocked_transform:
        mocked_transform.return_value = MagicMock(return_value=test_image)

        images = video_tensor_to_gift(tensor,
                                      duration=duration,
                                      loop=loop,
                                      optimize=optimize)

        mocked_transform.assert_called()
        test_image.save.assert_called_with(
            path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
            optimize=optimize,
        )


def test_video_tensor_to_gif_invalid_tensor():
    path = "/test/path"
    tensor = "invalid_tensor"

    with pytest.raises(TypeError):
        video_tensor_to_gift(tensor, path)


def test_video_tensor_to_gif_invalid_path():
    path = 123
    tensor = setup_test_tensor()

    with pytest.raises(TypeError):
        video_tensor_to_gift(tensor, path)


def test_video_tensor_to_gif_invalid_duration():
    path = "/test/path"
    tensor = setup_test_tensor()
    duration = "invalid_duration"

    with pytest.raises(TypeError):
        video_tensor_to_gift(tensor, path, duration=duration)


def test_video_tensor_to_gif_invalid_loop():
    path = "/test/path"
    tensor = setup_test_tensor()
    loop = "invalid_loop"

    with pytest.raises(TypeError):
        video_tensor_to_gift(tensor, path, loop=loop)
