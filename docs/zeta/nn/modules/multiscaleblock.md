# MultiScaleBlock

## **Table of Contents**

1. Overview
2. Class Definition
3. Functionality and Usage
4. Additional Tips & Information
5. Resources and References

## **1. Overview**

The `MultiScaleBlock` class, a component of PyTorch's `nn.Module`, falls under the category of deep learning models. PyTorch is a powerful, flexible deep learning framework that allows automatic differentiation and optimization. 

This class is well-suited to tasks where the spatial or temporal scale of the input data varies. Examples are wide-range in nature, including but not limited to, image processing, video analysis, and signal processing. 

In `MultiScaleBlock`, any PyTorch module such as convolutional layers, linear layers, or even sequence of layers can be applied to the input tensor at multiple scales in a seamless way. 

## **2. Class Definition**

### `MultiScaleBlock` Class 

The class definition for `MultiScaleBlock` is provided below:

```python
class MultiScaleBlock(nn.Module):
    """
    A module that applies a given submodule to the input tensor at multiple scales.

    Args:
        module (nn.Module): The submodule to be applied.

    Returns:
        torch.Tensor: The output tensor after applying the submodule at multiple scales.
    """

    def __init__(self, module):
        super().__init__()
        self.submodule = module

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x1 = F.interpolate(x, scale_factor=0.5, *args, **kwargs)
        x2 = F.interpolate(x, scale_factor=2.0, *args, **kwargs)
        return (
            self.submodule(x)
            + F.interpolate(self.submodule(x1), size=x.shape[2:])
            + F.interpolate(self.submodule(x2), size=x.shape[2:])
        )
```

#### Method 1: `__init__(self, module)`

This is the initializer for the `MultiScaleBlock` class, and it takes the following input:

- `module (nn.Module)`: The submodule to be applied on the input tensor at multiple scales.

#### Method 2: `forward(self, x: torch.Tensor, *args, **kwargs)`
The forward propagation method, onto which the initialized model is called with the input data `x`. It includes the following parameters:

- `x (torch.Tensor)`: The input tensor.
- `*args`: Additional arguments for the interpolate function of PyTorch. It can include various parameters depending on the Interpolation mode selected, which can be `mode`, `align_corners`, and `recompute_scale_factor`.
- `**kwargs`: Additional keyword arguments.

## **3. Functionality and Usage**

The `MultiScaleBlock` class is designed to apply a given submodule to the input tensor at multiple scales. The purpose of multi-scale processing is to handle the variation in scale of the different elements in the image, the data, or the signal.

In the `forward` method, the input tensor `x` is first interpolated at two different scales (0.5 and 2.0). The PyTorch function `torch.nn.functional.interpolate` adjusts the size of the tensor using specific scaling factors. Then, the submodule is applied to the original input tensor and the interpolated tensors. The output is the sum of the results of applying the submodule at the original scale and the two interpolated scales.

### **Usage Example**

Here are some examples showcasing the usage of `MultiScaleBlock`:

1. **Single Convolutional Layer as Submodule**:

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from zeta.nn import MultiScaleBlock

    conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    model = MultiScaleBlock(conv)
    input = torch.rand(1, 3, 32, 32)
    output = model(input)
    ```

2. **Sequence of Layers as Submodule**:

    ```python
    seq = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    model = MultiScaleBlock(seq)
    input = torch.rand(1, 3, 32, 32)
    output = model(input)
    ```

3. **Custom Model as Submodule**:

    Suppose `MyModel` is a PyTorch model, you can use `MultiScaleBlock` on it as follows:

    ```python
    model = MyModel(num_classes=10)
    multi_scale_model = MultiScaleBlock(model)
    input = torch.rand(1, 3, 32, 32)
    output = multi_scale_model(input)
    ```

## **4. Additional Information**

- The input tensor's shape must be in the form of (batch_size, num_channels, height, width) for `forward` method of this class to work properly. This is because the `F.interpolate` function in PyTorch expects the input in this format.

- This class uses `F.interpolate` function, make sure to check the PyTorch documentation for this function to understand various interpolation modes and their behavior: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html

## **5. References**

1. [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
2. [Multi-Scale Convolutional Neural Networks for Vision Tasks](https://arxiv.org/abs/1406.4729)

I hope this documentation will help you to understand and use `MultiScaleBlock` class in your scenarios. Enjoy DL with PyTorch!
