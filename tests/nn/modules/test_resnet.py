import pytest
import torch
from torch.nn import Conv2d

from zeta.nn.modules.res_net import ResNet


def test_resnet_init():
    resnet = ResNet(Conv2d, [2, 2, 2, 2])
    assert isinstance(resnet, ResNet)


def test_resnet_num_classes():
    resnet = ResNet(Conv2d, [2, 2, 2, 2], num_classes=10)
    assert resnet.fc.out_features == 10


def test_resnet_kernel_size():
    resnet = ResNet(Conv2d, [2, 2, 2, 2], kernel_size=5)
    assert resnet.conv1.kernel_size[0] == 5


def test_resnet_stride():
    resnet = ResNet(Conv2d, [2, 2, 2, 2], stride=3)
    assert resnet.conv1.stride[0] == 3


def test_resnet_block_type():
    with pytest.raises(TypeError):
        ResNet("not a block", [2, 2, 2, 2])


def test_resnet_num_blocks_not_list():
    with pytest.raises(TypeError):
        ResNet(Conv2d, "not a list")


def test_resnet_num_blocks_wrong_length():
    with pytest.raises(ValueError):
        ResNet(Conv2d, [2, 2, 2])


def test_resnet_num_blocks_not_integers():
    with pytest.raises(TypeError):
        ResNet(Conv2d, [2, 2, "not an integer", 2])


def test_resnet_forward():
    resnet = ResNet(Conv2d, [2, 2, 2, 2])
    x = torch.randn(1, 3, 224, 224)
    assert resnet(x).shape == torch.Size([1, 1000])


def test_resnet_forward_num_classes():
    resnet = ResNet(Conv2d, [2, 2, 2, 2], num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    assert resnet(x).shape == torch.Size([1, 10])


def test_resnet_forward_input_channels():
    resnet = ResNet(Conv2d, [2, 2, 2, 2])
    x = torch.randn(1, 1, 224, 224)
    with pytest.raises(RuntimeError):
        resnet(x)


def test_resnet_forward_input_size():
    resnet = ResNet(Conv2d, [2, 2, 2, 2])
    x = torch.randn(1, 3, 32, 32)
    with pytest.raises(RuntimeError):
        resnet(x)


def test_resnet_make_layer():
    resnet = ResNet(Conv2d, [2, 2, 2, 2])
    layer = resnet._make_layer(Conv2d, 64, 2, 1)
    assert isinstance(layer, torch.nn.Sequential)


def test_resnet_make_layer_block_type():
    resnet = ResNet(Conv2d, [2, 2, 2, 2])
    with pytest.raises(TypeError):
        resnet._make_layer("not a block", 64, 2, 1)


def test_resnet_make_layer_out_channels_not_integer():
    resnet = ResNet(Conv2d, [2, 2, 2, 2])
    with pytest.raises(TypeError):
        resnet._make_layer(Conv2d, "not an integer", 2, 1)


def test_resnet_make_layer_num_blocks_not_integer():
    resnet = ResNet(Conv2d, [2, 2, 2, 2])
    with pytest.raises(TypeError):
        resnet._make_layer(Conv2d, 64, "not an integer", 1)


def test_resnet_make_layer_stride_not_integer():
    resnet = ResNet(Conv2d, [2, 2, 2, 2])
    with pytest.raises(TypeError):
        resnet._make_layer(Conv2d, 64, 2, "not an integer")
