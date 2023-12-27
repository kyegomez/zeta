# save_load

# zeta.utils.save_load

## Description

The `save_load` function from the `zeta.utils` library defines a base decorator for both save and load methods for PyTorch's torch.nn.Module subclasses. This allows saving the state of a given module and configuration, and subsequently loading it back. This can be specifically useful when we want to store a trained model during the training process or at the end of it, and later resume training from where we left or use the trained model for inference. 

The decorator wraps the class initialization, saving, and loading methods. Additionally, optionally, it allows hook functions to be defined and executed right before saving and loading the model.

## Function Declaration

```python
def save_load(
    save_method_name: str = "save",
    load_method_name: str = "load",
    config_instance_var_name: str = "_config",
    init_and_load_classmethod_name: str = "init_and_load",
    version: Optional[str] = None,
    pre_save_hook: Optional[Callable[[Module], None]] = None,
    post_load_hook: Optional[Callable[[Module], None]] = None,
    compress: Optional[bool] = False,
    partial_load: Optional[bool] = False,
    *args,
    **kwargs,
):
```
## Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- | 
| `save_method_name` | str | Name of the save method. | `"save"` |
| `load_method_name` | str | Name of the load method. | `"load"` | 
| `config_instance_var_name` | str | Name of the instance variable to store the configuration. | `"_config"` | 
| `init_and_load_classmethod_name` | str | Name of the classmethod that initializes and loads the model. | `init_and_load` |
| `version` |str(optional) | Version of the model. | `None` | 
| `pre_save_hook` | Callable (optional) | This function is called before the model is saved. | `None` |
| `post_load_hook` | Callable (optional) | This function is called after the model is loaded | `None` |
| `compress` | bool (optional) | If True, uses the new zipfile-based TorchScript serialization format. | `False` |
| `partial_load` | bool(optional) | If
