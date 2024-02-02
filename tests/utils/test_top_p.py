# first, here are some imports and mock data setup:

import torch
import torch.nn.functional as F
import pytest
from zeta.utils import top_p

# mock data
logits = torch.FloatTensor([0.1, 0.2, 0.3, 0.4])
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
sorted_indices_to_remove = cum_probs > (1 - 0.9)


# Test if the return value is a tensor
def test_return_type():
    ret = top_p(logits)
    assert isinstance(ret, torch.Tensor)


# Test if the function is properly sorting the `logits`
def test_sorting():
    output = top_p(logits)
    assert torch.all(torch.eq(output, torch.sort(output, descending=True)[0]))


# Test if threshold argument is respected
def test_threshold():
    output = top_p(logits, thres=0.5)
    assert torch.cumsum(F.softmax(output, dim=-1), dim=-1)[-1].item() <= 0.5


# Test if the function is properly setting `-inf` for the values that should be removed
def test_inf_removal():
    top_p(logits)
    assert (sorted_logits[sorted_indices_to_remove] == float("-inf")).all()


# Test if function is properly scattering the results
def test_scattering():
    output = top_p(logits)
    assert torch.all(
        torch.eq(output, sorted_logits.scatter(1, sorted_indices,
                                               sorted_logits)))


# Test if the function is raising error for invalid `logits`
def test_invalid_logits():
    with pytest.raises(Exception):
        top_p(torch.Tensor([0.1, 0.2, None, 0.4]))


# Test if the function is raising error for invalid `thres`
def test_invalid_thres():
    with pytest.raises(Exception):
        top_p(logits, thres=1.5)
    with pytest.raises(Exception):
        top_p(logits, thres=-0.5)
