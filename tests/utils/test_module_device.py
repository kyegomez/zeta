import pytest
import torch
from torch.nn import Module
from zeta.utils.module_device import module_device


@module_device()
class DummyModule(Module):
    def __init__(self, x):
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor(x))


def test_module_device_init():
    module = DummyModule(5)
    assert isinstance(module, DummyModule)


def test_module_device_device_property():
    module = DummyModule(5)
    assert module.device == torch.device("cpu")


def test_module_device_to():
    module = DummyModule(5)
    module.to(torch.device("cpu"))
    assert module.device == torch.device("cpu")


def test_module_device_to_cuda():
    if torch.cuda.is_available():
        module = DummyModule(5)
        module.to(torch.device("cuda"))
        assert module.device == torch.device("cuda")


def test_module_device_to_cuda_compatibility_check():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):

            @module_device(compatibility_check=True)
            class IncompatibleModule(Module):
                def __init__(self, x):
                    super().__init__()
                    self.x = torch.nn.Parameter(torch.tensor(x))

            module = IncompatibleModule(5)
            module.to(torch.device("cuda"))


def test_module_device_device_property_name():
    @module_device(device_property_name="custom_device")
    class CustomDeviceModule(Module):
        def __init__(self, x):
            super().__init__()
            self.x = torch.nn.Parameter(torch.tensor(x))

    module = CustomDeviceModule(5)
    assert module.custom_device == torch.device("cpu")


def test_module_device_not_module():
    with pytest.raises(AssertionError):

        @module_device()
        class NotAModule:
            pass


def test_module_device_multiple_devices():
    if torch.cuda.is_available():

        @module_device()
        class MultiDeviceModule(Module):
            def __init__(self, x):
                super().__init__()
                self.x = torch.nn.Parameter(torch.tensor(x))
                self.y = torch.nn.Parameter(
                    torch.tensor(x), device=torch.device("cuda")
                )

        module = MultiDeviceModule(5)
        assert len(module.device) > 1
