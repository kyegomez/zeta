import pytest
import torch
from torch import nn
from zeta.nn.modules.dynamic_module import DynamicModule

def test_dynamicmodule_initialization():
    model = DynamicModule()
    assert isinstance(model, DynamicModule)
    assert model.module_dict == nn.ModuleDict()
    assert model.forward_method == None

def test_dynamicmodule_add_remove_module():
    model = DynamicModule()
    model.add('linear', nn.Linear(10, 10))
    assert 'linear' in model.module_dict
    model.remove('linear')
    assert 'linear' not in model.module_dict

def test_dynamicmodule_forward():
    model = DynamicModule()
    model.add('linear', nn.Linear(10, 10))
    x = torch.randn(1, 10)
    output = model(x)
    assert output.shape == (1, 10)

@pytest.mark.parametrize("name", ['linear'])
def test_dynamicmodule_add_module_edge_cases(name):
    model = DynamicModule()
    model.add(name, nn.Linear(10, 10))
    with pytest.raises(Exception):
        model.add(name, nn.Linear(10, 10))

@pytest.mark.parametrize("name", ['linear'])
def test_dynamicmodule_remove_module_edge_cases(name):
    model = DynamicModule()
    with pytest.raises(Exception):
        model.remove(name)