# Module Documentation: `module_device`

## Overview

The `module_device` module provides a powerful decorator for PyTorch neural network modules that allows you to manage and control the device on which a module and its associated parameters reside. This decorator simplifies the management of device transfers, making it easier to ensure your model runs on the desired hardware.

This documentation will guide you through the `module_device` decorator's architecture, purpose, functions, and usage examples. You'll learn how to effectively use this decorator to control the device placement of your PyTorch modules.

## Table of Contents

1. [Installation](#installation)
2. [Architecture](#architecture)
3. [Purpose](#purpose)
4. [Decorator: module_device](#decorator-module_device)
    - [Parameters](#parameters)
    - [Usage Examples](#usage-examples)
        - [Basic Usage](#basic-usage)
        - [Custom Device Property Name](#custom-device-property-name)
        - [On Device Transfer Callback](#on-device-transfer-callback)
5. [Additional Information](#additional-information)
6. [References](#references)

---

## 1. Installation <a name="installation"></a>

The `module_device` decorator is a Python code snippet that can be directly incorporated into your project without the need for separate installation.

## 2. Architecture <a name="architecture"></a>

The `module_device` decorator is a Python decorator that can be applied to subclasses of PyTorch's `nn.Module`. It adds device management capabilities to your modules by providing control over the device on which a module and its parameters reside.

## 3. Purpose <a name="purpose"></a>

The primary purpose of the `module_device` decorator is to simplify the management of device transfers for PyTorch neural network modules. It allows you to specify the target device, handle compatibility checks, and execute callbacks when transferring a module to a different device.

## 4. Decorator: module_device <a name="decorator-module_device"></a>

The `module_device` decorator provides the following functionality:

- Device management: Control the device on which a module and its parameters reside.
- Custom device property name: Define a custom property name for accessing the module's current device.
- On device transfer callback: Execute a custom callback when transferring a module to a different device.

### Parameters <a name="parameters"></a>

The `module_device` decorator accepts the following parameters:

- `device_property_name` (str, optional): The name of the property that will be used to access the module's current device. Defaults to "device".
- `on_device_transfer` (Callable, optional): A callback function that is executed when transferring the module to a different device. Defaults to None.
- `compatibility_check` (bool, optional): Enable or disable compatibility checks for device transfers. Defaults to False.

### Usage Examples <a name="usage-examples"></a>

#### Basic Usage <a name="basic-usage"></a>

Here's a basic example of using the `module_device` decorator to manage the device of a PyTorch module:

```python
import torch
from torch.nn import Module
from zeta.utils import module_device

@module_device()
class MyModule(Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.fc = torch.nn.Linear(10, 5)

# Create an instance of MyModule
my_model = MyModule()

# Access the device property
print(my_model.device)  # This will print the device of the module
```

#### Custom Device Property Name <a name="custom-device-property-name"></a>

You can define a custom device property name when using the `module_device` decorator:

```python
import torch
from torch.nn import Module
from zeta.utils import module_device

@module_device(device_property_name="custom_device")
class CustomModule(Module):
    def __init__(self):
        super(CustomModule, self).__init__()
        self.fc = torch.nn.Linear(10, 5)

# Create an instance of CustomModule
custom_model = CustomModule()

# Access the custom device property
print(custom_model.custom_device)
```

#### On Device Transfer Callback <a name="on-device-transfer-callback"></a>

You can specify a callback function to be executed when transferring a module to a different device:

```python
import torch
from torch.nn import Module
from zeta.utils import module_device

def on_device_transfer_callback(module, device):
    print(f"Transferred to {device}")

@module_device(on_device_transfer=on_device_transfer_callback)
class CallbackModule(Module):
    def __init__(self):
        super(CallbackModule, self).__init__()
        self.fc = torch.nn.Linear(10, 5)

# Create an instance of CallbackModule
callback_model = CallbackModule()

# Transfer the model to a different device
callback_model.to(torch.device("cuda:0"))
```

## 5. Additional Information <a name="additional-information"></a>

- The `module_device` decorator simplifies device management for PyTorch modules, allowing you to focus on your model's functionality.
- Compatibility checks can be enabled to ensure that device transfers are compatible with the available hardware.
- Callbacks provide a way to execute custom actions when transferring a module to a different device.

## 6. References <a name="references"></a>

For more information on PyTorch and device management, refer to the official PyTorch documentation: [PyTorch Device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device).

