import torch
from torch.nn import Module


def module_device(
    device_property_name: str = "device",
    on_device_transfer=None,
    compatibility_check: bool = False,
):
    """Module device decorator.

    Args:
        device_property_name (str, optional): _description_. Defaults to "device".
        on_device_transfer (_type_, optional): _description_. Defaults to None.
        compatibility_check (bool, optional): _description_. Defaults to False.
    """

    def decorator(klass):
        assert issubclass(
            klass, Module
        ), "should decorate a subclass of torch.nn.Module"

        _orig_init = klass.__init__
        _orig_to = klass.to

        def __init__(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self.register_buffer("_dummy", torch.tensor(0), persistent=False)

        def __to(self, device, *args, **kwargs):
            if (
                compatibility_check
                and not torch.cuda.is_available()
                and "cuda" in str(device)
            ):
                raise RuntimeError(
                    "CUDA is not available for this device transfer."
                )
            result = _orig_to(self, device, *args, **kwargs)
            if on_device_transfer:
                on_device_transfer(self, device)
            return result

        @property
        def _device_property(self):
            devices = {p.device for p in self.parameters()} | {
                b.device for b in self.buffers()
            }
            if len(devices) > 1:
                return devices
            return self._dummy.device

        klass.__init__ = __init__
        klass.to = __to
        setattr(klass, device_property_name, _device_property)

        return klass

    return decorator
