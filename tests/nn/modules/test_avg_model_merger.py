import torch
import torch.nn as nn
from zeta.nn.modules.avg_model_merger import AverageModelMerger


def test_average_model_merger_init():
    model1 = nn.Linear(10, 10)
    model2 = nn.Linear(10, 10)
    merger = AverageModelMerger([model1, model2])
    assert isinstance(merger, AverageModelMerger)
    assert merger.models == [model1, model2]


def test_average_model_merger_merge_models():
    model1 = nn.Linear(10, 10)
    model2 = nn.Linear(10, 10)
    merger = AverageModelMerger([model1, model2])
    merged_model = merger.merge_models()
    assert isinstance(merged_model, nn.Module)
    assert merged_model.state_dict().keys() == model1.state_dict().keys()


def test_average_model_merger_copy_model_structure():
    model = nn.Linear(10, 10)
    merger = AverageModelMerger([model])
    model_copy = merger._copy_model_structure(model)
    assert isinstance(model_copy, nn.Module)
    assert model_copy.state_dict().keys() == model.state_dict().keys()


def test_average_model_merger_merge_models_weights():
    model1 = nn.Linear(10, 10)
    model2 = nn.Linear(10, 10)
    merger = AverageModelMerger([model1, model2])
    merged_model = merger.merge_models()
    for param_tensor in merged_model.state_dict():
        assert torch.allclose(
            merged_model.state_dict()[param_tensor],
            (model1.state_dict()[param_tensor] +
             model2.state_dict()[param_tensor]) / 2,
        )
