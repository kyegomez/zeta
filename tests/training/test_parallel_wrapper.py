from unittest.mock import MagicMock, patch

import torch.nn as nn

from zeta.training.parallel_wrapper import ParallelWrapper


# Test initialization
def test_init():
    model = nn.Linear(512, 512)
    wrapper = ParallelWrapper(model)

    assert wrapper.model == model
    assert wrapper.device == "cuda"
    assert wrapper.use_data_parallel is True


# Test forward method
def test_forward():
    model = nn.Linear(512, 512)
    wrapper = ParallelWrapper(model)

    # Mock the forward method of the model
    model.forward = MagicMock(return_value="forward result")
    result = wrapper.forward("input")

    model.forward.assert_called_once_with("input")
    assert result == "forward result"


# Test to method
def test_to():
    model = nn.Linear(512, 512)
    wrapper = ParallelWrapper(model)

    # Mock the to method of the model
    model.to = MagicMock(return_value=model)
    wrapper = wrapper.to("cpu")

    model.to.assert_called_once_with("cpu")
    assert wrapper.device == "cpu"


# Test __getattr__ method
def test_getattr():
    model = nn.Linear(512, 512)
    wrapper = ParallelWrapper(model)

    assert wrapper.in_features == model.in_features
    assert wrapper.out_features == model.out_features


# Test data parallelism
@patch("torch.cuda.device_count", return_value=2)
def test_data_parallelism(mocked_device_count):
    model = nn.Linear(512, 512)
    wrapper = ParallelWrapper(model)

    mocked_device_count.assert_called_once()
    assert isinstance(wrapper.model, nn.DataParallel)


# Test data parallelism with single GPU
@patch("torch.cuda.device_count", return_value=1)
def test_data_parallelism_single_gpu(mocked_device_count):
    model = nn.Linear(512, 512)
    wrapper = ParallelWrapper(model, use_data_parallel=True)

    mocked_device_count.assert_called_once()
    assert not isinstance(wrapper.model, nn.DataParallel)
