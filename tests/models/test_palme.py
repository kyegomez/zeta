import pytest
import torch

from zeta.models import PalmE
from zeta.structs import AutoregressiveWrapper, ViTransformerWrapper


@pytest.fixture
def palme():
    return PalmE(image_size=128, patch_size=16, num_tokens=5)


def test_palme_initialization(palme):
    assert isinstance(palme, PalmE)
    assert isinstance(palme.encoder, ViTransformerWrapper)
    assert isinstance(palme.decoder, AutoregressiveWrapper)
    assert palme.decoder_dim == 512


def test_palme_forward(palme):
    # Prepare the test input
    img = torch.rand(1, 3, 128, 128)
    text = torch.randint(5, (1, 1))

    # Try normal forward pass
    output = palme(img, text)
    assert isinstance(output, torch.Tensor)


def test_palme_forward_raise_exception(palme):
    with pytest.raises(Exception) as e:
        # Pass in bad inputs to trigger exception
        bad_img, bad_text = "not an image", "not a text"
        palme(bad_img, bad_text)

    assert "Failed in forward method" in str(e)
