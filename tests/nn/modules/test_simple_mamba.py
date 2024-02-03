import torch
from torch import nn

from zeta.nn.modules.simple_mamba import (
    Mamba,
    MambaBlock,
    RMSNorm,
)


def test_mamba_class_init():
    model = Mamba(10000, 512, 6)

    assert isinstance(model.embedding, nn.Embedding)
    assert isinstance(model.layers, nn.ModuleList)
    assert isinstance(model.norm_f, RMSNorm)
    assert isinstance(model.lm_head, nn.Linear)


def test_mamba_forward():
    model = Mamba(10000, 512, 6)
    x = torch.randint(0, 10000, (1, 50))
    out = model(x)

    assert out.shape == torch.Size([1, 50, 10000])


def test_mamba_different_vocab_size():
    model = Mamba(20000, 512, 6)
    x = torch.randint(0, 20000, (1, 50))
    out = model(x)

    assert out.shape == torch.Size([1, 50, 20000])


def test_mamba_different_dim():
    model = Mamba(10000, 1024, 6)
    x = torch.randint(0, 10000, (1, 50))
    out = model(x)

    assert out.shape == torch.Size([1, 50, 10000])


def test_mamba_different_depth():
    model = Mamba(10000, 512, 12)
    x = torch.randint(0, 10000, (1, 50))
    out = model(x)

    assert out.shape == torch.Size([1, 50, 10000])


def test_mamba_with_dropout():
    model = Mamba(10000, 512, 6, dropout=0.5)
    x = torch.randint(0, 10000, (1, 50))
    out = model(x)

    assert out.shape == torch.Size([1, 50, 10000])


def test_mamba_with_custom_layer():
    class CustomLayer(nn.Module):
        def forward(self, x):
            return x * 2

    model = Mamba(10000, 512, 6, layer=CustomLayer())
    x = torch.randint(0, 10000, (1, 50))
    out = model(x)

    assert out.shape == torch.Size([1, 50, 10000])


def test_mamba_block_class_init():
    block = MambaBlock(dim=64, depth=1)

    assert isinstance(block.in_proj, nn.Linear)
    assert isinstance(block.conv1d, nn.Conv1d)
    assert isinstance(block.x_proj, nn.Linear)
    assert isinstance(block.dt_proj, nn.Linear)
    assert isinstance(block.out_proj, nn.Linear)


def test_mamba_block_forward():
    block = MambaBlock(dim=64, depth=1)
    x = torch.randn(1, 10, 64)
    out = block(x)

    assert out.shape == torch.Size([1, 10, 64])


def test_mamba_block_different_dim():
    block = MambaBlock(dim=128, depth=1)
    x = torch.randn(1, 10, 128)
    out = block(x)

    assert out.shape == torch.Size([1, 10, 128])


def test_mamba_block_different_depth():
    block = MambaBlock(dim=64, depth=2)
    x = torch.randn(1, 10, 64)
    out = block(x)

    assert out.shape == torch.Size([1, 10, 64])


def test_mamba_block_with_custom_dim_inner():
    block = MambaBlock(dim=64, dim_inner=128, depth=1)
    x = torch.randn(1, 10, 64)
    out = block(x)

    assert out.shape == torch.Size([1, 10, 64])


def test_mamba_block_with_custom_d_state():
    block = MambaBlock(dim=64, d_state=32, depth=1)
    x = torch.randn(1, 10, 64)
    out = block(x)

    assert out.shape == torch.Size([1, 10, 64])


def test_mamba_block_with_custom_expand():
    block = MambaBlock(dim=64, expand=3, depth=1)
    x = torch.randn(1, 10, 64)
    out = block(x)

    assert out.shape == torch.Size([1, 10, 64])


def test_mamba_block_with_custom_dt_rank():
    block = MambaBlock(dim=64, dt_rank=10, depth=1)
    x = torch.randn(1, 10, 64)
    out = block(x)

    assert out.shape == torch.Size([1, 10, 64])


def test_mamba_block_with_custom_d_conv():
    block = MambaBlock(dim=64, d_conv=8, depth=1)
    x = torch.randn(1, 10, 64)
    out = block(x)

    assert out.shape == torch.Size([1, 10, 64])


def test_mamba_block_with_custom_conv_bias():
    block = MambaBlock(dim=64, conv_bias=False, depth=1)
    x = torch.randn(1, 10, 64)
    out = block(x)

    assert out.shape == torch.Size([1, 10, 64])


def test_mamba_block_with_custom_bias():
    block = MambaBlock(dim=64, bias=True, depth=1)
    x = torch.randn(1, 10, 64)
    out = block(x)

    assert out.shape == torch.Size([1, 10, 64])
