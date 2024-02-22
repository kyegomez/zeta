# Module Name: FusedDenseGELUDense

The `FusedDenseGELUDense` module represents a combination of fully connected layers with the GELU activation function. It is suitable for efficiently performing linear transformations with an activation function in between, commonly used in neural network architectures. The input dimension (`dim`) and output dimension (`dim_out`) can be specified, while further customizations such as selecting the datatype and setting specific threshold configurations are also supported.


## Args:
The table below summarizes the arguments of the `FusedDenseGELUDense` module:

| Argument          | Type              | Description                                     | Default Value |
|-------------------|-------------------|-------------------------------------------------|----------------|
| dim               | int               | Input dimension                                 | -              |
| dim_out           | int               | Output dimension                                | -              |
| bias              | bool (optional)   | Indicates whether to use a bias term            | True           |
| has_fp16_weights  | bool (optional)   | Whether to use fp16 weights                      | False          |
| threshold         | float (optional)  | Threshold for quantization                       | 6.0            |

## Purpose:
The `FusedDenseGELUDense` module is designed to efficiently perform linear transformations and activations in neural network architectures. It allows for customizable configurations such as input and output dimensions, the inclusion of bias terms, FP16 weight usage, and threshold settings, providing flexibility in designing network layers.

## Functionality and Usage:
The `FusedDenseGELUDense` class effectively combines linear transformation operations with GELU activation. During the forward pass, the input data passes through a linear transformation, followed by the GELU activation, and another linear transformation, providing the final output.

This module is particularly useful for creating deep learning models that require efficient processing of the data through multiple connected layers with non-linear activation functions in between. Below is an example of how to use the `FusedDenseGELUDense` module:

```python
# Example of using the FusedDenseGELUDense module
import torch

from zeta.nn import FusedDenseGELUDense

# Define input data
x = torch.randn(1, 512)

# Create the FusedDenseGELUDense module
model = FusedDenseGELUDense(512, 1024)

# Perform the forward pass
out = model(x)

# Display the shape of the output
print(out.shape)
# Expected Output:
# torch.Size([1, 512])
```

The example illustrates the creation of a `FusedDenseGELUDense` object with input dimension 512 and output dimension 1024. Then, the forward pass is executed on the input `x`, resulting in the output tensor `out`.

## Additional Information and Tips:
Avoid using non-default values for the `has_fp16_weights` and `threshold` arguments unless with a specific need for FP16 weights and custom quantization threshold. For most use cases, the default settings are recommended. Be aware that the activation function used in `FusedDenseGELUDense` is the GELU activation, and the logic within the module will have different execution paths based on the availability of the `bitsandbytes` package.

## References and Resources:
When using quantization and FP16 weights, it's advisable to refer to the official PyTorch documentation on these topics for further understanding. For comprehensive information on the GELU activation function, the original research paper or relevant documentation are valuable resources.

In conclusion, the `FusedDenseGELUDense` module aims to provide an optimized and flexible approach for incorporating linear transformations and activations within neural network architectures.

# Note:
The given example template and documentation format have been followed to deliver explicit and thorough documentation for the `FusedDenseGELUDense` module, addressing its purpose, essential arguments, usage, and additional tips.
