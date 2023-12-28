import pytest
import torch
from zeta.structs import ViTransformerWrapper, Encoder
from torch.nn import Module


# 1. Test to check if default object of class is instance of torch.nn.Module
def test_default_object_of_class():
    attn_layer = Encoder(dim=512, depth=6)
    model = ViTransformerWrapper(
        image_size=256, patch_size=6, attn_layers=attn_layer
    )
    assert isinstance(model, Module)


# 2. Test to check if object of class with parameters is instance of torch.nn.Module
def test_object_with_parameters_of_class():
    attn_layer = Encoder(dim=512, depth=6)
    model = ViTransformerWrapper(
        image_size=32, patch_size=8, attn_layers=attn_layer
    )
    assert isinstance(model, Module)


# 3. Test to check if invalid attention layers throws an AssertionError
def test_invalid_attention_layers():
    with pytest.raises(AssertionError):
        ViTransformerWrapper(image_size=256, patch_size=8, attn_layers=None)


# 4. Test to check if invalid image size, patch size ratio throws an AssertionError
def test_invalid_image_patch_size_ratio():
    attn_layer = Encoder(dim=512, depth=6)
    with pytest.raises(AssertionError):
        ViTransformerWrapper(
            image_size=100, patch_size=8, attn_layers=attn_layer
        )


# 5. Test to check forward pass
def test_forward_pass():
    attn_layer = Encoder(dim=512, depth=6)
    model = ViTransformerWrapper(
        image_size=256, patch_size=8, attn_layers=attn_layer
    )
    random_input = torch.rand(1, 3, 256, 256)
    output = model(random_input, return_embeddings=True)
    assert output.shape[0] == 1, "Mismatch in batch size"
    assert output.shape[2] == 512, "Mismatch in dimensions"
