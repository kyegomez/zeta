# Module Documentation: `save_load`

## Overview

The `save_load` module provides a powerful decorator for PyTorch neural network modules that simplifies the process of saving and loading model checkpoints. This decorator is designed to enhance the ease and flexibility of managing model checkpoints, making it more efficient to work with PyTorch models during development and production.

This documentation will guide you through the `save_load` decorator's architecture, purpose, functions, and usage examples. You'll learn how to effectively use this decorator to save and load model checkpoints, manage configuration settings, and handle version compatibility.

## Table of Contents

1. [Installation](#installation)
2. [Architecture](#architecture)
3. [Purpose](#purpose)
4. [Decorator: save_load](#decorator-save_load)
    - [Parameters](#parameters)
    - [Usage Examples](#usage-examples)
        - [Basic Usage](#basic-usage)
        - [Custom Methods and Hooks](#custom-methods-and-hooks)
        - [Partial Loading](#partial-loading)
        - [Version Compatibility](#version-compatibility)
5. [Additional Information](#additional-information)
6. [References](#references)

---

## 1. Installation <a name="installation"></a>

The `save_load` decorator is a Python code snippet that can be directly incorporated into your project without the need for separate installation.

## 2. Architecture <a name="architecture"></a>

The `save_load` decorator is a Python decorator that can be applied to subclasses of PyTorch's `nn.Module`. It enhances the module with methods for saving and loading model checkpoints, including options for configuration management, version compatibility, and custom hooks.

## 3. Purpose <a name="purpose"></a>

The primary purpose of the `save_load` decorator is to streamline the process of saving and loading PyTorch model checkpoints. It offers the following benefits:

- Simplified checkpoint management: Provides easy-to-use methods for saving and loading model states.
- Configuration preservation: Allows for the preservation and retrieval of the module's configuration settings.
- Version compatibility: Offers mechanisms to handle version compatibility between saved checkpoints.
- Customization: Supports custom hooks that can be executed before and after saving or loading.

## 4. Decorator: save_load <a name="decorator-save_load"></a>

The `save_load` decorator provides the following functionality:

- Saving and loading model checkpoints.
- Configuration preservation: Saving and retrieving configuration settings.
- Version compatibility: Checking and handling version mismatches.
- Customization: Executing custom hooks before and after saving or loading.

### Parameters <a name="parameters"></a>

The `save_load` decorator accepts the following parameters:

- `save_method_name` (str, optional): The name of the method used for saving the model checkpoint. Defaults to "save".
- `load_method_name` (str, optional): The name of the method used for loading the model checkpoint. Defaults to "load".
- `config_instance_var_name` (str, optional): The name of the instance variable used to store the configuration. Defaults to "_config".
- `init_and_load_classmethod_name` (str, optional): The name of the class method used to initialize and load a model from a checkpoint. Defaults to "init_and_load".
- `version` (Optional[str], optional): The version of the saved checkpoint. Defaults to None.
- `pre_save_hook` (Optional[Callable[[Module], None]], optional): A callback function executed before saving the model checkpoint. Defaults to None.
- `post_load_hook` (Optional[Callable[[Module], None]], optional): A callback function executed after loading the model checkpoint. Defaults to None.
- `compress` (Optional[bool], optional): Enable compression when saving checkpoints. Defaults to False.
- `partial_load` (Optional[bool], optional): Enable partial loading of the model checkpoint. Defaults to False.

### Usage Examples <a name="usage-examples"></a>

#### Basic Usage <a name="basic-usage"></a>

Here's a basic example of using the `save_load` decorator to save and load a PyTorch model checkpoint:

```python
import torch
from torch.nn import Module
from zeta.utils import save_load

@save_load()
class MyModel(Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = torch.nn.Linear(10, 5)

# Create an instance of MyModel
my_model = MyModel()

# Save the model checkpoint
my_model.save("my_model.pth")

# Load the model checkpoint
loaded_model = MyModel.load("my_model.pth")
```

#### Custom Methods and Hooks <a name="custom-methods-and-hooks"></a>

You can define custom method and hook names when using the `save_load` decorator:

```python
import torch
from torch.nn import Module
from zeta.utils import save_load

@save_load(
    save_method_name="custom_save",
    load_method_name="custom_load",
    pre_save_hook=my_pre_save_hook,
    post_load_hook=my_post_load_hook
)
class CustomModel(Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.fc = torch.nn.Linear(10, 5)

# Create an instance of CustomModel
custom_model = CustomModel()

# Custom save and load
custom_model.custom_save("custom_model.pth")
loaded_custom_model = CustomModel.custom_load("custom_model.pth")
```

#### Partial Loading <a name="partial-loading"></a>

Enable partial loading to update only specific parts of the model checkpoint:

```python
import torch
from torch.nn import Module
from zeta.utils import save_load

@save_load(partial_load=True)
class PartialModel(Module):
    def __init__(self):
        super(PartialModel, self).__init__()
        self.fc = torch.nn.Linear(10, 5)

# Create an instance of PartialModel
partial_model = PartialModel()

# Save the model checkpoint
partial_model.save("partial_model.pth")

# Load only the updated part of the model checkpoint
loaded_partial_model = PartialModel.load("partial_model.pth")
```

#### Version Compatibility <a name="version-compatibility"></a>

Handle version compatibility when loading saved checkpoints:

```python
import torch
from torch.nn import Module
from zeta.utils import save_load

@save_load(version="1.0")
class VersionedModel(Module):
    def __init__(self):
        super(VersionedModel, self).__init__()
        self.fc = torch.nn.Linear(10, 5)

# Create an instance of VersionedModel
versioned_model = VersionedModel()

# Save the model checkpoint
versioned_model.save("versioned_model.pth")

# Load the model checkpoint with version compatibility check
loaded_versioned_model = VersionedModel.load("versioned_model.pth")
```

## 5. Additional Information <a name="additional-information"></a>

- The `save_load` decorator simplifies the process of saving and loading model checkpoints for PyTorch modules.
- Configuration settings can be preserved and retrieved along with the model checkpoint.
- Version compatibility checks help manage saved checkpoints with different versions.
- Custom hooks can be used to execute custom actions before and after saving or loading checkpoints.

## 6. References <a name="references"></a>

For more information on PyTorch and checkpoint management, refer to the official PyTorch documentation: [PyTorch

 Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

