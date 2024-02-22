# FILEPATH: /Users/defalt/Desktop/Athena/research/zeta/tests/nn/modules/test_img_patch_embed.py

import torch
from torch import nn

from zeta.nn.modules.img_patch_embed import ImgPatchEmbed


def test_class_init():
    model = ImgPatchEmbed()

    assert isinstance(model.proj, nn.Conv2d)
    assert model.img_size == 224
    assert model.patch_size == 16
    assert model.num_patches == 196


def test_class_init_with_args():
    model = ImgPatchEmbed(
        img_size=448, patch_size=32, in_chans=1, embed_dim=512
    )

    assert isinstance(model.proj, nn.Conv2d)
    assert model.img_size == 448
    assert model.patch_size == 32
    assert model.num_patches == 196
    assert model.proj.in_channels == 1
    assert model.proj.out_channels == 512


def test_forward():
    model = ImgPatchEmbed()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)

    assert out.shape == torch.Size([1, 196, 768])


def test_forward_with_different_input():
    model = ImgPatchEmbed()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert out.shape == torch.Size([2, 196, 768])


def test_forward_with_different_img_size():
    model = ImgPatchEmbed(img_size=448)
    x = torch.randn(1, 3, 448, 448)
    out = model(x)

    assert out.shape == torch.Size([1, 196, 768])


def test_forward_with_different_patch_size():
    model = ImgPatchEmbed(patch_size=32)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)

    assert out.shape == torch.Size([1, 49, 768])


def test_forward_with_different_in_chans():
    model = ImgPatchEmbed(in_chans=1)
    x = torch.randn(1, 1, 224, 224)
    out = model(x)

    assert out.shape == torch.Size([1, 196, 768])


def test_forward_with_different_embed_dim():
    model = ImgPatchEmbed(embed_dim=512)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)

    assert out.shape == torch.Size([1, 196, 512])
