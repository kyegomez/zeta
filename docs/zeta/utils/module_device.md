# module_device

# Module Name: module_device

The `module_device` is a Python decorator function that efficiently manages a device on which a PyTorch neural network models, which is a subclass of `torch.nn.Module`, is loaded. This decorator helps in tracking the device on which different components (such as tensors) of the model are, especially in complex design models where different tensors can be on separate devices. This helps to avoid any device mismatch errors during computation.

Moreover, it allows the developers to add their custom functions or operations that could be performed whenever the device changes. Also, it has an in-built compatibility check feature, which elegantly handles the case of trying to transfer to GPUs when CUDA is not available.

To dive deep, let's see the main components and details of this function.

## Class Defintion:
```python
def module_device(
    device_property_name: str = "device",
    on_device_transfer=None,
    compatibility_check: bool = False,
):
```
This function has three parameters – `device_property_name`, `on_device_transfer`, and `compatibility_check`.

| Parameter              | Type   |  Default  |  Description                                                                                                                                 |
|------------------------|--------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------|
| device_property_name   | string |  "device" | Name of the attribute which would track the device of the decorated class.                                                                  |
| on_device_transfer     | callable/disable | None   | A callable function that will be invoked whenever the device changes. This function will be executed after the object is transferred to a new device. If None, no function will be executed. |
| compatibility_check    | boolean    | False   | If True, checks the compatibility of the device change in case of CUDA not being available when trying to transfer to GPUs.   |

Here, `_dummy` is a registered buffer, a PyTorch state that is not a parametric tensor of the model but you want to save the model, so it persists across saving/loading roundtrips.

In case of multiple GPUs and your model spans them, this decorator will store all the devices.

The `decorator` function wraps around a user-defined class. It keeps track of the device and throws an error when an incompatible device is used and updates the new device property in case of valid device change. It can also assist in performing user defined operations in case of device change using `on_device_transfer` function.

## Usage Examples:
Let's look at three ways to use this function.

### Example 1:
In the first example, we simply use this decorator to add a new device property (named "my_cuda_device" here) to our model, which always stores the current device of our model.

```python
from torch import tensor
from torch.nn import Module


@module_device(device_property_name="my_cuda_device")
class MyModel(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)


MyModel_obj = MyModel(10, 10)
MyModel_obj.to("cuda")

print(MyModel_obj.my_cuda_device)  # Output: cuda:<device_no>
```
### Example 2:

In the second example, we will define a function that will be executed whenever the device changes. Here for simplicity, we will just print a simple message.

```python
def transfer_fn(self, device):
    print(f"Transferred to {device}")


@module_device(on_device_transfer=transfer_fn)
class SecondModel(Module):
    pass


SecondModel_obj = SecondModel()
SecondModel_obj.to("cuda")  # Output: Transferred to cuda:<device_no>
```
### Example 3:

In the third example, we will use both the features discussed above together:

```python
def transfer_fn(self, device):
    print(f"Transferred to {device}")


@module_device(device_property_name="my_device", on_device_transfer=transfer_fn)
class ThirdModel(Module):
    pass


ThirdModel_obj = ThirdModel()
ThirdModel_obj.to("cuda")  # Output: Transferred to cuda:<device_no>
print(ThirdModel_obj.my_device)  # Output: cuda:<device_no>
```
