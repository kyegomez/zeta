import pytest
import torch
from zeta.utils.cuda_memory_wrapper import track_cuda_memory_usage


def test_track_cuda_memory_usage_no_cuda():

    @track_cuda_memory_usage
    def test_func():
        return "Hello, World!"

    assert test_func() == "Hello, World!"


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA is not available")
def test_track_cuda_memory_usage_with_cuda():

    @track_cuda_memory_usage
    def test_func():
        return torch.tensor([1, 2, 3]).cuda()

    assert torch.equal(test_func(), torch.tensor([1, 2, 3]).cuda())


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA is not available")
def test_track_cuda_memory_usage_with_cuda_memory_allocation():

    @track_cuda_memory_usage
    def test_func():
        a = torch.tensor([1, 2, 3]).cuda()
        b = torch.tensor([4, 5, 6]).cuda()
        return a + b

    assert torch.equal(test_func(), torch.tensor([5, 7, 9]).cuda())


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA is not available")
def test_track_cuda_memory_usage_with_cuda_memory_release():

    @track_cuda_memory_usage
    def test_func():
        a = torch.tensor([1, 2, 3]).cuda()
        b = torch.tensor([4, 5, 6]).cuda()
        del a
        del b
        torch.cuda.empty_cache()

    assert test_func() is None


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA is not available")
def test_track_cuda_memory_usage_with_exception():

    @track_cuda_memory_usage
    def test_func():
        a = torch.tensor([1, 2, 3]).cuda()
        b = "not a tensor"
        return a + b

    with pytest.raises(TypeError):
        test_func()
