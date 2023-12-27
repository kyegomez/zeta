import pytest
from torch.nn import Module
import torch

from zeta.utils.module_device import module_device


class TestModule(Module):
    pass


@module_device("device", compatibility_check=True)
class CompatibleModule(Module):
    pass


@module_device("device", on_device_transfer=lambda self, device: None)
class OnTransferModule(Module):
    pass


def test_module_device_with_compatibility_check():
    test_module = CompatibleModule()

    # device - str
    if torch.cuda.is_available():
        assert test_module.to("cuda") == test_module
    else:
        with pytest.raises(RuntimeError):
            test_module.to("cuda")

    # device - torch.device
    if torch.cuda.is_available():
        assert test_module.to(torch.device("cuda")) == test_module
    else:
        with pytest.raises(RuntimeError):
            test_module.to(torch.device("cuda"))


def test_on_device_transfer_functionality():
    test_module = OnTransferModule()

    # on_device_transfer should be called when transferred without raising any exception
    # more extensive tests could be done depending on the implementation of on_device_transfer
    assert test_module.to("cpu") == test_module


def test_module_device_without_decorator():
    test_module = TestModule()

    # without decorator, transfer should go through without any issues
    assert test_module.to("cpu") == test_module
    if torch.cuda.is_available():
        assert test_module.to("cuda") == test_module


def test_device_property():
    test_module = TestModule()

    # without decorator, there should be no 'device' property
    with pytest.raises(AttributeError):
        test_module.device

    # with decorator, 'device' property should exist
    test_module = CompatibleModule()
    assert isinstance(test_module.device, torch.device)
