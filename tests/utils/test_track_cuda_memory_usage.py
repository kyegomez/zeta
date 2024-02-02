import pytest
from unittest.mock import patch
from zeta.utils import track_cuda_memory_usage


# Testing the base functionality with cuda available and function without error
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.memory_allocated", side_effect=[1000, 2000])
@patch("torch.cuda.synchronize")
@patch("logging.info")
def test_track_cuda_memory_usage_base(mock_log_info, mock_sync, mock_mem_alloc,
                                      mock_cuda_avail):

    @track_cuda_memory_usage
    def test_func():
        return "Test"

    assert test_func() == "Test"
    mock_sync.assert_called()
    mock_mem_alloc.assert_called()
    mock_log_info.assert_called_with("Memory usage of test_func: 1000 bytes")


# Testing function with an exception
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.memory_allocated", side_effect=[1000, 2000])
@patch("torch.cuda.synchronize")
@patch("logging.info")
def test_track_cuda_memory_usage_exception(mock_log_info, mock_sync,
                                           mock_mem_alloc, mock_cuda_avail):

    @track_cuda_memory_usage
    def test_func():
        raise ValueError("Test exception")

    with pytest.raises(ValueError):
        test_func()

    mock_sync.assert_called()
    mock_mem_alloc.assert_called()
    mock_log_info.assert_called_with("Memory usage of test_func: 1000 bytes")


# Testing when cuda is not available
@patch("torch.cuda.is_available", return_value=False)
@patch("torch.cuda.memory_allocated")
@patch("torch.cuda.synchronize")
@patch("logging.warning")
def test_track_cuda_memory_usage_no_cuda(mock_log_warn, mock_sync,
                                         mock_mem_alloc, mock_cuda_avail):

    @track_cuda_memory_usage
    def test_func():
        return "Test"

    assert test_func() == "Test"
    mock_sync.assert_not_called()
    mock_mem_alloc.assert_not_called()
    mock_log_warn.assert_called_with(
        "CUDA is not available, skip tracking memory usage")
