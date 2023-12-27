# DualPathBlock


**Table of Contents**

1. [Introduction](#introduction)
2. [Key Features](#features)
3. [Class Definition](#class-definition)
4. [Example Usage](#examples)
5. [Practical Tips](#tips)
6. [Reference and Other Resources](#resources)

## Introduction <a id="introduction"></a>
The `DualPathBlock` class is a PyTorch-based module or grammar that represents a basic computational unit in dual path networks. This class combines the output of two submodules by element-wise addition. The core idea behind this method is to efficiently use the information from both paths in a balanced way.

## Key Features <a id="features"></a>

- **Efficient combination of data**: The `DualPathBlock` method combines data from two submodules in an effective way by using element-wise addition.

- **Flexibility in submodule choice**: Users have the flexibility to choose the submodules, provided they are `torch.nn.Module` instances.

- **Simplicity and readability of code**: Due to its modular design, the code is easy to understand, thereby making it easier for users to implement and modify.

- **Easy integration with other `torch.nn.Module` instances**: The `DualPathBlock` can be easily integrated within other pipelines as a subnet.

## Class Definition <a id="class-definition"></a>

The class design for `DualPathBlock` is very straightforward. It is initialized with two submodules that are instances of `nn.Module`. Then, during the forward pass, the inputs are passed through each submodule and the result of these computations is then computed by element-wise addition.

### Parameters:

|Parameter|Type|Description|
|---|---|---|
|submodule1|nn.Module|First submodule through which input tensor `x` is passed.|
|submodule2|nn.Module|Second submodule through which input tensor `x` is passed.|

### Methods:

|Method|Parameters|Description|
|---|---|---|
|forward|x: torch.Tensor|Performs forward pass through the model. Calculates output tensor obtained by adding outputs of submodule1 and submodule2. Returns the computed tensor|

### Input / Output Type:

- **Input**: Receives a tensor of any shape.
- **Output**: Produces a tensor of the same shape as the inputs after the forward computation is done.

## Example Usage <a id="examples"></a>

```python
# Import the necessary libraries
import torch
import torch.nn as nn
from zeta.nn import DualPathBlock

# Define two simple submodule
submodule1 = nn.Linear(20, 20)
submodule2 = nn.Linear(20, 20)

# Create an instance of DualPathBlock
dual_path_block = DualPathBlock(submodule1, submodule2)

# Define an input tensor
input_tensor = torch.randn(10, 20)

# Perform forward operation
output = dual_path_block(input_tensor)

# Print the output tensor
print(output)
```
## Practical Tips <a id="tips"></a>

- While DualPathBlock design allows for the use of any submodules, please make sure the outputs of both submodules can be summed up i.e., they are of the same shape.

- DualPathBlock is particularly useful in constructing networks with parallel paths where the outputs are combined. 

## References and Other Resources <a id="resources"></a>
[Pytorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

[Dual Path Networks](https://arxiv.org/abs/1707.01629) <-- If relevant

