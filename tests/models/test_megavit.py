import pytest
import torch

from zeta.models import MegaVit

# Basic tests, checking instantiation and forward pass with different parameters


def test_MegaVit_instantiation():
    model = MegaVit(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1,
    )
    assert isinstance(model, MegaVit)


def test_MegaVit_forward_pass():
    model = MegaVit(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1,
    )
    img = torch.randn(1, 3, 256, 256)
    result = model(img)
    assert result.shape == (1, 1000)


# Parameterized tests with different input (checking for compatibility with different sized images)


@pytest.mark.parametrize("img_size", [128, 256, 512])
def test_MegaVit_with_different_image_sizes(img_size):
    model = MegaVit(
        image_size=img_size,
        patch_size=32,
        num_classes=1000,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1,
    )
    img = torch.randn(1, 3, img_size, img_size)
    result = model(img)
    assert result.shape == (1, 1000)


# Exception tests


def test_blank_image_MegaVit():
    model = MegaVit(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1,
    )
    img = torch.zeros(1, 3, 256, 256)
    with pytest.raises(Exception):
        model(img)
