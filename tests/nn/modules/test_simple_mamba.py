# FILEPATH: /Users/defalt/Desktop/Athena/research/zeta/tests/nn/modules/test_simple_mamba.py

import pytest
import torch
from torch import nn
from zeta.nn.modules.simple_mamba import Mamba, ResidualBlock, RMSNorm


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


def test_residual_block_class_init():
    block = ResidualBlock(512)

    assert isinstance(block.norm1, RMSNorm)
    assert isinstance(block.norm2, RMSNorm)
    assert isinstance(block.fc1, nn.Linear)
    assert isinstance(block.fc2, nn.Linear)


def test_residual_block_forward():
    block = ResidualBlock(512)
    x = torch.randn(1, 50, 512)
    out = block(x)

    assert out.shape == torch.Size([1, 50, 512])


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


def test_residual_block_different_dim():
    block = ResidualBlock(1024)
    x = torch.randn(1, 50, 1024)
    out = block(x)

    assert out.shape == torch.Size([1, 50, 1024])


def test_mamba_with_dropout():
    model = Mamba(10000, 512, 6, dropout=0.5)
    x = torch.randint(0, 10000, (1, 50))
    out = model(x)

    assert out.shape == torch.Size([1, 50, 10000])


def test_residual_block_with_dropout():
    block = ResidualBlock(512, dropout=0.5)
    x = torch.randn(1, 50, 512)
    out = block(x)

    assert out.shape == torch.Size([1, 50, 512])


def test_mamba_with_custom_layer():
    class CustomLayer(nn.Module):
        def forward(self, x):
            return x * 2

    model = Mamba(10000, 512, 6, layer=CustomLayer())
    x = torch.randint(0, 10000, (1, 50))
    out = model(x)

    assert out.shape == torch.Size([1, 50, 10000])
