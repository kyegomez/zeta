# save_load

# zeta.utils.save_load 

## Overview

The `save_load` decorator in the `zeta.utils` module is a Python decorator designed around PyTorch's `torch.nn.Module` subclasses. Its main functionality is to automate and streamline the saving and loading of trained models and their configurations, reducing the need for repeated code and increasing code readability and maintainability.

Key to its purpose is the ability to handle the model's state dictionary, training configurations, and PyTorch version. The decorator enhances the training workflow by allowing models’ states and configurations to be easily saved and loaded efficiently with built-in version compatibility checks and hooks for code execution pre and post-saving/loading.

## Core Functionality

### save_load Decorator

Considered a Base decorator for save and load methods for `torch.nn.Module` subclasses. In essence, a decorator is a higher-order function that can drape functionality over other functions or classes without changing their source code, which is exactly what the `save_load` decorator is.

The `save_load` decorator modifies `torch.nn.Module` subclasses by adding save, load and an initialization & load methods to the subclass. This allows for seamless saving and loading of the subclass instances states and configurations.

## Function / Method definition 

```
@beartype
def save_load(
    save_method_name="save",
    load_method_name="load",
    config_instance_var_name="_config",
    init_and_load_classmethod_name="init_and_load",
    version: Optional[str] = None,
    pre_save_hook: Optional[Callable[[Module], None]] = None,
    post_load_hook: Optional[Callable[[Module], None]] = None,
    compress: Optional[bool] = False,
    partial_load: Optional[bool] = False,
    *args,
    **kwargs,
):...
```

The function takes in several arguments:

| Parameter               | Type                             | Default               | Description                                                                                            |
|-------------------------|----------------------------------|-----------------------|--------------------------------------------------------------------------------------------------------|
| `save_method_name`      | `str`                            | `"save"`              | The name used to set the save method for the instance.                                                 |
| `load_method_name`      | `str`                            | `"load"`              | The name used to set the load method for the instance.                                                 |
| `config_instance_var_name`| `str`                          | `"_config"`           | The name used to set the instance's configuration variable.                                            |
| `init_and_load_classmethod_name`| `str`                    | `"init_and_load"`     | The name used to set the class's initialization and loading method.                                    |
| `version`               | `Optional[str]`                  | `None`                | Version of the torch module. Used for checking compatibility when loading.                              |
| `pre_save_hook`         | `Optional[Callable[[Module], None]]`| `None`             | Callback function before saving. Useful for final operations before saving states and configurations.  |
| `post_load_hook`        | `Optional[Callable[[Module], None]]`| `None`             | Callback function after loading. Ideal for any additional operations after loading states and configurations. |
| `compress`              | `Optional[bool]`                 | `False`               | If set to `True`, the saved model checkpoints will be compressed.                                       |
| `partial_load`          | `Optional[bool]`                 | `False`               | If set to `True`, the saved model checkpoint will be partially loaded to existing models.               |
| `*args` & `**kwargs`    | `Any`                            |                       | Additional arguments for the decorator.                                                                 |


The *save_load* decorator modifies the way a PyTorch model is initialized, saved, and loaded. It does this by wrapping new init, save, load, and init_and_load methods around the decorated class.

## Usage Examples

Here is a basic usage example of the `save_load` decorator:

### Example 1:  Using default parameters on a PyTorch Model
```python
from torch.nn import Linear, Module

from zeta.utils import save_load


@save_load()
class MyModel(Module):

    def __init__(self, dim, output_dim):
        super().__init__()
        self.layer = Linear(dim, output_dim)

    def forward(self, x):
        return self.layer(x)


# Initialize your model
model = MyModel(32, 10)

# Save your model
model.save("model.pt")

# Load your model
loadedim = MyModel.load("model.pt")
```

### Example 2:  Using the `save_load` with non-default arguments
In this example, we are going to add `pre_save_hook` and `post_load_hook` to demonstrate their usage. These functions will be called just before saving and
