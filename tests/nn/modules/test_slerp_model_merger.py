import torch
import torch.nn as nn

from zeta.nn.modules.slerp_model_merger import SLERPModelMerger


def test_slerp_model_merger_init():
    model1 = nn.Linear(10, 10)
    model2 = nn.Linear(10, 10)
    merger = SLERPModelMerger(model1, model2, 0.5)
    assert isinstance(merger, SLERPModelMerger)
    assert merger.t == 0.5
    assert merger.model1 is model1
    assert merger.model2 is model2


def test_slerp_model_merger_merge():
    model1 = nn.Linear(10, 10)
    model2 = nn.Linear(10, 10)
    merger = SLERPModelMerger(model1, model2, 0.5)
    mergedim = merger.merge()
    assert isinstance(mergedim, nn.Module)
    assert mergedim.state_dict().keys() == model1.state_dict().keys()


def test_slerp_model_merger_slerp():
    model1 = nn.Linear(10, 10)
    model2 = nn.Linear(10, 10)
    merger = SLERPModelMerger(model1, model2, 0.5)
    w1 = torch.randn(10)
    w2 = torch.randn(10)
    t = 0.5
    slerp_result = merger._slerp(w1, w2, t)
    assert slerp_result.shape == w1.shape


def test_slerp_model_merger_copy_model_structure():
    model1 = nn.Linear(10, 10)
    model2 = nn.Linear(10, 10)
    merger = SLERPModelMerger(model1, model2, 0.5)
    model_copy = merger._copy_model_structure(model1)
    assert isinstance(model_copy, nn.Module)
    assert model_copy.state_dict().keys() == model1.state_dict().keys()
