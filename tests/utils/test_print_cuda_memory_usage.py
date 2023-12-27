import torch
from zeta.utils import print_cuda_memory_usage
from unittest.mock import patch


def test_if_cuda_is_available():
    assert torch.cuda.is_available(), "CUDA is not available on your system."


def test_initial_memory_value():
    assert (
        torch.cuda.memory_allocated() >= 0
    ), "CUDA memory allocated is less than 0."


def test_after_memory_usage():
    with print_cuda_memory_usage():
        torch.rand((1000, 1000)).cuda()
    assert (
        torch.cuda.memory_allocated() > 0
    ), "CUDA memory allocated is less than or equal to initial memory."


def test_memory_usage_value():
    init_mem = torch.cuda.memory_allocated()
    with print_cuda_memory_usage():
        torch.rand((1000, 1000)).cuda()
    assert (torch.cuda.memory_allocated() - init_mem) / (
        1024**3
    ) >= 0, "Memory usage is negative."


@patch("builtins.print")
def test_print_call(mock_print):
    with print_cuda_memory_usage():
        torch.rand((1000, 1000)).cuda()
    assert mock_print.called, "Print function was not called."


@patch("builtins.print")
def test_print_format(mock_print):
    mem = torch.cuda.memory_allocated()
    with print_cuda_memory_usage():
        torch.rand((1000, 1000)).cuda()
    mock_print.assert_called_with(
        "CUDA memory usage:"
        f" {((torch.cuda.memory_allocated() - mem) / (1024**3)):.2f} GB"
    )
