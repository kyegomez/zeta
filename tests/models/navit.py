import pytest
import torch
from zeta.models import NaViT
from torch.nn.modules.module import ModuleAttributeError
from torch.nn import Sequential


# ---- SETUP ----
@pytest.fixture
def neural_network_template():
    model = NaViT(
        image_size=100,
        patch_size=10,
        num_classes=2,
        dim=100,
        depth=2,
        heads=2,
        mlp_dim=2,
    )
    return model


# ---- TESTS ----


# Verify if the model is an instance of nn.Module
def test_model_instantiation(neural_network_template):
    assert isinstance(neural_network_template, NaViT)


# Test the forward method
def test_forward_method(neural_network_template):
    input_tensor = torch.ones([10, 3, 100, 100])
    result = neural_network_template(input_tensor)
    assert result.is_cuda
    assert result.requires_grad


# Test the dropout configuration
def test_dropout_configuration(neural_network_template):
    assert neural_network_template.dropout.p == 0.0


# Test the proper initialisation of LayerNorm and Linear layers
def test_layers_initialization(neural_network_template):
    sequence = neural_network_template.to_patch_embedding
    assert isinstance(sequence, Sequential)
    assert len(sequence) == 3


# Test if the transformer is properly initialised
def test_transformer_initialization(neural_network_template):
    assert neural_network_template.transformer.dim == 100


# Test the device property
def test_device_property(neural_network_template):
    assert str(neural_network_template.device).startswith("cuda")


# Test if the dimensions of the input image are correct
def test_if_model_raises_error_on_wrong_dimensions(neural_network_template):
    input_tensor = torch.ones([10, 3, 50, 50])
    with pytest.raises(AssertionError):
        _ = neural_network_template(input_tensor)


# Test the behaviour when token_dropout_prob is an int or a float
def test_token_dropout(neural_network_template):
    model = neural_network_template
    model.token_dropout_prob = 0.5
    assert callable(model.calc_token_dropout)


# Test if exceptions are thrown when they should be
def test_exceptions(neural_network_template):
    with pytest.raises(ModuleAttributeError):
        _ = neural_network_template.non_existent_attribute


# add your test cases here..
