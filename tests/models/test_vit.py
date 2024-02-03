import torch
import pytest
from zeta.models import ViT
from zeta.structs import Encoder

# Sample Tests


def test_initialization():
    attn_layers = Encoder(...)
    model = ViT(image_size=256, patch_size=32, attn_layers=attn_layers)
    assert model.patch_size == 32
    assert isinstance(model.pos_embedding, torch.nn.Parameter)
    assert isinstance(model.patch_to_embedding, torch.nn.Sequential)
    assert isinstance(model.dropout, torch.nn.Dropout)
    assert isinstance(model.attn_layers, Encoder)


def test_forward():
    attn_layers = Encoder(...)
    model = ViT(image_size=256, patch_size=32, attn_layers=attn_layers)
    img = torch.rand(1, 3, 256, 256)
    x = model.forward(img)
    assert x.shape == (1, attn_layers.dim)  # Expected output shape


def test_invalid_type_attn_layers():
    attn_layers = "DummyEncoder"
    with pytest.raises(AssertionError):
        ViT(image_size=256, patch_size=32, attn_layers=attn_layers)


def test_invalid_size():
    attn_layers = Encoder(...)
    # An image size that's not divisible by patch size
    with pytest.raises(AssertionError):
        ViT(image_size=257, patch_size=32, attn_layers=attn_layers)


@pytest.mark.parametrize(
    "image_size, patch_size", [(256, 32), (512, 64), (1024, 128), (2048, 256)]
)
def test_varied_sizes(image_size, patch_size):
    attn_layers = Encoder(...)
    model = ViT(
        image_size=image_size, patch_size=patch_size, attn_layers=attn_layers
    )
    img = torch.rand(1, 3, image_size, image_size)
    x = model.forward(img)
    assert x.shape == (1, attn_layers.dim)


# further tests are created using the same pattern for each attribute/method/edge condition
