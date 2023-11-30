import pytest
import torch
from zeta.nn.embeddings.vision_emb import VisionEmbedding


def test_visionembedding_initialization():
    model = VisionEmbedding(
        img_size=224, patch_size=16, in_chans=3, embed_dim=768
    )
    assert isinstance(model, VisionEmbedding)
    assert model.img_size == (224, 224)
    assert model.patch_size == (16, 16)
    assert model.num_patches == 196
    assert model.proj.kernel_size == (16, 16)


def test_visionembedding_forward():
    model = VisionEmbedding(
        img_size=224, patch_size=16, in_chans=3, embed_dim=768
    )
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    assert output.shape == (1, 197, 768)


@pytest.mark.parametrize("img_size", [0])
def test_visionembedding_forward_edge_cases(img_size):
    model = VisionEmbedding(
        img_size=img_size, patch_size=16, in_chans=3, embed_dim=768
    )
    x = torch.randn(1, 3, img_size, img_size)
    with pytest.raises(Exception):
        model(x)


def test_visionembedding_forward_invalid_dimensions():
    model = VisionEmbedding(
        img_size=224, patch_size=16, in_chans=3, embed_dim=768
    )
    x = torch.randn(1, 3, 128, 128)
    with pytest.raises(Exception):
        model(x)


# Test case for default initialization
def test_default_init():
    module = VisionEmbedding()
    assert module.img_size == (224, 224)
    assert module.patch_size == (16, 16)
    assert module.num_patches == 197
    assert isinstance(module.proj, torch.nn.Conv2d)
    assert module.mask_token is None
    assert module.cls_token is None


# Test case for custom initialization
def test_custom_init():
    module = VisionEmbedding(
        img_size=128,
        patch_size=32,
        in_chans=1,
        embed_dim=512,
        contain_mask_token=True,
        prepend_cls_token=True,
    )
    assert module.img_size == (128, 128)
    assert module.patch_size == (32, 32)
    assert module.num_patches == 16
    assert isinstance(module.proj, torch.nn.Conv2d)
    assert module.mask_token is not None
    assert module.cls_token is not None


# Test case for forward pass with default settings
def test_forward_default():
    module = VisionEmbedding()
    x = torch.randn(2, 3, 224, 224)
    y = module(x)
    assert y.shape == (2, 197, 768)


# Test case for forward pass with custom settings
def test_forward_custom():
    module = VisionEmbedding(
        img_size=128,
        patch_size=32,
        in_chans=1,
        embed_dim=512,
        contain_mask_token=True,
        prepend_cls_token=True,
    )
    x = torch.randn(2, 1, 128, 128)
    masked_position = torch.randint(0, 2, (2, 17))
    y = module(x, masked_position)
    assert y.shape == (2, 18, 512)


# Test case for initializing with incorrect image size
def test_incorrect_img_size_init():
    with pytest.raises(AssertionError):
        module = VisionEmbedding(img_size=256)


# Test case for initializing with incorrect patch size
def test_incorrect_patch_size_init():
    with pytest.raises(AssertionError):
        module = VisionEmbedding(patch_size=64)


# Test case for initializing with negative in_chans
def test_negative_in_chans_init():
    with pytest.raises(ValueError):
        module = VisionEmbedding(in_chans=-3)


# Test case for initializing with negative embed_dim
def test_negative_embed_dim_init():
    with pytest.raises(ValueError):
        module = VisionEmbedding(embed_dim=-768)


# Test case for initializing with invalid masked_position
def test_invalid_masked_position_init():
    module = VisionEmbedding(contain_mask_token=True)
    with pytest.raises(AssertionError):
        x = torch.randn(2, 3, 224, 224)
        masked_position = torch.randint(0, 2, (2, 17))
        module(x, masked_position)


# Test case for initializing with invalid cls_token
def test_invalid_cls_token_init():
    module = VisionEmbedding(prepend_cls_token=True)
    with pytest.raises(AssertionError):
        x = torch.randn(2, 3, 224, 224)
        module(x)


# Test case for num_position_embeddings
def test_num_position_embeddings():
    module = VisionEmbedding()
    assert module.num_position_embeddings() == 197


# Test case for forward pass with mask token
def test_forward_mask_token():
    module = VisionEmbedding(contain_mask_token=True)
    x = torch.randn(2, 3, 224, 224)
    masked_position = torch.randint(0, 2, (2, 197))
    y = module(x, masked_position)
    assert y.shape == (2, 197, 768)


# Test case for forward pass with cls token
def test_forward_cls_token():
    module = VisionEmbedding(prepend_cls_token=True)
    x = torch.randn(2, 3, 224, 224)
    y = module(x)
    assert y.shape == (2, 198, 768)


# Test case for forward pass with both mask and cls tokens
def test_forward_mask_and_cls_tokens():
    module = VisionEmbedding(contain_mask_token=True, prepend_cls_token=True)
    x = torch.randn(2, 3, 224, 224)
    masked_position = torch.randint(0, 2, (2, 197))
    y = module(x, masked_position)
    assert y.shape == (2, 198, 768)
