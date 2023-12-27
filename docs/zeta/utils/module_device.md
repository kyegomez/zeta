# module_device

# Module Name: module_device

This decorator provides an extended functionality to PyTorch's nn.Module. PyTorch's nn.Module does not have a specific property that explicitly points out which device it resides on. This decorator provides the `device` property to the class that can be used to return the device of a particular PyTorch's nn.Module class.

## Function Definition

The decorator is defined as follows:

```python
def module_device(
    device_property_name: str = "device",
    on_device_transfer=None,
    compatibility_check: bool = False,
):
```

### Parameters

| Parameter              | Type    | Default Value | Description |
|------------------------|---------|---------------|-------------|
| device_property_name   | str     | "device"        | The name of the device property. |
| on_device_transfer     | function| None            | A function to be called whenever the device is transferred.|
| compatibility_check    | bool    | False           | If set to True, raises an exception if "cuda" is in the device string while CUDA is not available. |

## Inner Functions and Properties

### decorator

```python
def decorator(klass):
```
The function takes a class as input and then checks if the input `klass` is a subclass of torch.nn.Module.

### \_\_init\_\_

```python
def __init__(self, *args, **kwargs):
```
It overrides the original `__init__` method of the class and registers a buffer named "_dummy", which is a non-persistent tensor containing a single zero. 

### \_\_to

```python
def __to(self, device, *args, **kwargs):
```
This function is overloading the `to()` method of the torch.nn.Module class. It first checks if the `compatibility_check` flag is true and CUDA is not available, but the device is "cuda". If this is the case, a RuntimeError is raised. Otherwise, the `to()` method of torch.nn.Module is called with the specified parameters.

### _device_property

```python
@property
def _device_property(self):
```
The `_device_property` helps in fetching the device property of the object. It does not take any parameters and returns the device on which the model is residing. It does this by checking the device of all parameters and buffers of the model. if the model resides on more than one device, it returns all the
