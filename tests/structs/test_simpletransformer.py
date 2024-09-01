import pytest
import torch
import torch.nn as nn

from zeta.structs import SimpleTransformer


def test_valid_init():
    """Test initialization of SimpleTransformer."""
    stm = SimpleTransformer(512, 6, 20_000)
    assert isinstance(stm, SimpleTransformer)
    assert isinstance(stm.emb, nn.Embedding)
    assert isinstance(stm.to_logits, nn.Sequential)


def test_forward_output_shape():
    """Test forward method of SimpleTransformer."""
    stm = SimpleTransformer(512, 6, 20_000)
    x = torch.randn(2, 1024).long()
    y = stm(x)
    assert y.shape == torch.Size([2, 1024, 20_000])


@pytest.mark.parametrize(
    "x_arg", [(32.2), (["str1", "str2"]), (512, 6, "20000")]
)
def test_invalid_forward_input_raises_error(x_arg):
    """Test forward method raises ValueError with invalid input."""
    stm = SimpleTransformer(512, 6, 20_000)
    with pytest.raises((TypeError, ValueError)):
        stm(x_arg)
