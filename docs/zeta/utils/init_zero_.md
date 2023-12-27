# init_zero_

# Module Name: zeta.utils

## Function Name: init_zero_

The `init_zero_` function is used to initialize the weights and bias of a PyTorch layer to zero. Initialization of the weights and biases of a layer play a crucial role regarding the performance of a deep learning model. Here, we're initializing every parameter to zero, turning the model into a "zero model". This is useful for certain tasks where you need your model to start with a clean slate.

This function is designed to work with any layer type available in the `torch.nn.Module` of PyTorch framework. However, it should be noted that if we initialize parameters of all layers as zero, then all the neurons at each layer will learn the same features during training. This function should be used when you're sure that initializing parameters to zero fits your specific needs.

Below is the function definition and description of the parameters:

| Function parameters | Description                                                                                                        |
|---------------------|--------------------------------------------------------------------------------------------------------------------|
| layer               |A `torch.nn.Module` object: The layer to initialize.|

```python
def init_zero_(layer):
    """
    Initialize the weights and bias of a torch layer to zero.

    Args:
        layer (torch.nn.Module): The layer to initialize.
    """
    nn.init.constant_(layer.weight, 0.0)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)
```

## How to Use init_zero_

Below we provide three different examples showing the usage of `init_zero_` function.

### Example 1: Initializing a Linear Layer with `init_zero_`

```python
import torch.nn as nn
import zeta.utils as utils

# define a linear layer
linear_layer = nn.Linear(10, 5)

# initialize the layer with zeros
utils.init_zero_(linear_layer)

# print the weights and the bias of the layer
print(linear_layer.weight)
print(linear_layer.bias)
```

### Example 2: Initializing a Convolutional Layer with `init_zero_`

```python
import torch.nn as nn
import zeta.utils as utils

# define a 2d convolutional layer
conv_layer = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

# initialize the layer with zeros
utils.init_zero_(conv_layer)

# print the weights and the bias of the layer

