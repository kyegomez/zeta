import pytest
import torch
from zeta.nn.embeddings.vision_emb import VisionEmbedding

def test_visionembedding_initialization():
    model = VisionEmbedding(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
    assert isinstance(model, VisionEmbedding)
    assert model.img_size == (224, 224)
    assert model.patch_size == (16, 16)
    assert model.num_patches == 196
    assert model.proj.kernel_size == (16, 16)

def test_visionembedding_forward():
    model = VisionEmbedding(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    assert output.shape == (1, 197, 768)

@pytest.mark.parametrize("img_size", [0])
def test_visionembedding_forward_edge_cases(img_size):
    model = VisionEmbedding(img_size=img_size, patch_size=16, in_chans=3, embed_dim=768)
    x = torch.randn(1, 3, img_size, img_size)
    with pytest.raises(Exception):
        model(x)

def test_visionembedding_forward_invalid_dimensions():
    model = VisionEmbedding(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
    x = torch.randn(1, 3, 128, 128)
    with pytest.raises(Exception):
        model(x)