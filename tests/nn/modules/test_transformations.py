import pytest
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomResizedCrop,
    Resize,
    CenterCrop,
)
from zeta.nn.modules.transformations import (
    image_transform,
    _convert_to_rgb,
    ToTensor,
    ResizeMaxSize,
    F,
)


# Define some fixtures for common parameters
@pytest.fixture
def image_size():
    return 256


@pytest.fixture
def is_train():
    return True


@pytest.fixture
def mean():
    return (0.48145466, 0.4578275, 0.40821073)


@pytest.fixture
def std():
    return (0.26862954, 0.26130258, 0.27577711)


@pytest.fixture
def resize_longest_max():
    return False


@pytest.fixture
def fill_color():
    return 0


@pytest.fixture
def inmem():
    return False


# Test the function with default parameters
def test_image_transform_defaults(image_size, is_train, mean, std):
    transform = image_transform(image_size, is_train)
    assert isinstance(transform, Compose)
    assert len(transform.transforms) == 4
    assert isinstance(transform.transforms[0], RandomResizedCrop)
    assert transform.transforms[1] == _convert_to_rgb
    assert isinstance(transform.transforms[2], ToTensor)
    assert isinstance(transform.transforms[3], Normalize)
    assert transform.transforms[3].mean == mean
    assert transform.transforms[3].std == std


# Test the function with custom parameters
def test_image_transform_custom(
    image_size, is_train, mean, std, resize_longest_max, fill_color
):
    transform = image_transform(
        image_size, is_train, mean, std, resize_longest_max, fill_color
    )
    assert isinstance(transform, Compose)
    assert len(transform.transforms) == 5
    assert isinstance(transform.transforms[0], Resize)
    assert isinstance(transform.transforms[1], CenterCrop)
    assert transform.transforms[2] == _convert_to_rgb
    assert isinstance(transform.transforms[3], ToTensor)
    assert isinstance(transform.transforms[4], Normalize)
    assert transform.transforms[4].mean == mean
    assert transform.transforms[4].std == std


# Test the function with inmem parameter
def test_image_transform_inmem(image_size, is_train, mean, std, inmem):
    transform = image_transform(image_size, is_train, mean, std, inmem=inmem)
    assert isinstance(transform, Compose)
    assert len(transform.transforms) == 3
    assert isinstance(transform.transforms[0], RandomResizedCrop)
    assert transform.transforms[1] == _convert_to_rgb
    assert transform.transforms[2] == F.pil_to_tensor


# Test the function with resize_longest_max parameter
def test_image_transform_resize_longest_max(
    image_size, is_train, mean, std, resize_longest_max
):
    transform = image_transform(
        image_size, is_train, mean, std, resize_longest_max=resize_longest_max
    )
    assert isinstance(transform, Compose)
    assert len(transform.transforms) == 4
    assert isinstance(transform.transforms[0], ResizeMaxSize)
    assert transform.transforms[1] == _convert_to_rgb
    assert isinstance(transform.transforms[2], ToTensor)
    assert isinstance(transform.transforms[3], Normalize)
    assert transform.transforms[3].mean == mean
    assert transform.transforms[3].std == std
