# `MixtureOfSoftmaxes` Documentation

The `MixtureOfSoftmaxes` module is an implementation of the Mixture of Softmaxes (MoS) as described by Yang et al. in 2017. This module enhances the expressiveness of the softmax function by combining multiple softmaxes. It is particularly useful for tasks where the relationship between input features and output classes is complex and can benefit from a combination of multiple softmax distributions.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Initialization](#initialization)
  - [Forward Pass](#forward-pass)
- [Examples](#examples)
  - [Basic Example](#basic-example)
  - [Complex Task](#complex-task)
- [Parameters](#parameters)
- [Return Value](#return-value)
- [Additional Information](#additional-information)
- [References](#references)

## Overview <a name="overview"></a>

The `MixtureOfSoftmaxes` module is designed to improve the modeling capabilities of the softmax function by allowing the combination of multiple softmax distributions. It takes an input tensor and computes a weighted sum of softmax outputs from different softmax layers. These weights are learned during training, enabling the model to adapt to the data's characteristics effectively.

The primary use case of the MoS module is in scenarios where a single softmax may not capture the complex relationships between input features and output classes. By combining multiple softmax distributions with learned mixture weights, the module provides a flexible approach to handle such situations.

## Installation <a name="installation"></a>

Before using the `MixtureOfSoftmaxes` module, ensure you have the required dependencies installed. You'll need:

- zetascale

You can install Zeta using pip:

```bash
pip install zetascale
```

Once you have the dependencies installed, you can import the module in your Python code.

```python
import torch
from torch import nn
from zeta.ops import MixtureOfSoftmaxes
```

## Usage <a name="usage"></a>

### Initialization <a name="initialization"></a>

To use the `MixtureOfSoftmaxes` module, you need to create an instance of it by providing the following arguments during initialization:

- `num_mixtures` (int): The number of softmax mixtures.
- `input_size` (int): The size of the input feature dimension.
- `num_classes` (int): The number of classes in the output dimension.

Here's an example of how to initialize the module:

```python
mos = MixtureOfSoftmaxes(num_mixtures=5, input_size=128, num_classes=10)
```

### Forward Pass <a name="forward-pass"></a>

Once you've initialized the `MixtureOfSoftmaxes` module, you can perform the forward pass by passing an input tensor `x` to it. The forward pass calculates the combined output from the mixture of softmaxes.

```python
x = torch.randn(32, 128)  # Example input tensor
output = mos(x)
```

The `output` tensor will contain the combined result from the mixture of softmax distributions.

## Examples <a name="examples"></a>

### Basic Example <a name="basic-example"></a>

Here's a simple example of how to use the `MixtureOfSoftmaxes` module to handle a classification task:

```python
import torch
from torch import nn
from zeta.ops import MixtureOfSoftmaxes


# Initialize the module
mos = MixtureOfSoftmaxes(num_mixtures=3, input_size=128, num_classes=10)

# Generate random input data
x = torch.randn(32, 128)

# Perform the forward pass
output = mos(x)

print(output.shape)  # Expected output shape: torch.Size([32, 10])
```

In this example, we create an instance of `MixtureOfSoftmaxes` with three mixtures, an input size of 128, and ten output classes. We then generate random input data and perform a forward pass to get the output.

### Complex Task <a name="complex-task"></a>

In more complex scenarios, the MoS module can be applied to tasks where traditional softmax may not be sufficient. For example, in natural language processing (NLP), the MoS module can be used to model complex relationships between words and their meanings.

```python
import torch
from torch import nn
from zeta.ops import MixtureOfSoftmaxes

# Initialize the module
mos = MixtureOfSoftmaxes(num_mixtures=5, input_size=128, num_classes=10000)  # Large vocabulary size

# Generate input data (word embeddings)
x = torch.randn(32, 128)

# Perform the forward pass
output = mos(x)

print(output.shape)  # Expected output shape: torch.Size([32, 10000])
```

In this example, we initialize the MoS module with five mixtures and a large vocabulary size (10,000 classes). This demonstrates the module's ability to handle complex tasks with a significant number of output classes.

## Parameters <a name="parameters"></a>

Here are the parameters that can be passed during the initialization of the `MixtureOfSoftmaxes` module:

| Parameter            | Description                                                | Data Type | Default Value |
|----------------------|------------------------------------------------------------|-----------|---------------|
| `num_mixtures`       | Number of softmax mixtures.                                | int       | -             |
| `input_size`         | Size of the input feature dimension.                       | int       | -             |
| `num_classes`        | Number of classes in the output dimension.                 | int       | -             |

## Return Value <a name="return-value"></a>

The `forward` method of the `MixtureOfSoftmaxes` module returns two values:

1. `attn_output` (Tensor): The combined output from the mixture of softmaxes.
2. `attn_output_weights` (Optional[Tensor]): The attention weights. Only returned when `need_weights` is set to `True`.

## Additional Information <a name="additional-information"></a>

- The MoS module can be used in a variety of deep learning tasks, including classification, natural language processing, and more.
- It is important to fine-tune the number of mixtures and other hyperparameters based on the specific task and dataset.

## References <a name="references"></a>

- Yang, Z., Hu, Z., Salakhutdinov, R., and Berg-Kirkpatrick, T. (2017). Improved variational inference with inverse autoregressive flow. In Proceedings of the 34th International Conference on Machine Learning (ICML).

This documentation provides a comprehensive guide on using the `MixtureOfSoftmaxes` module. Feel free to explore its capabilities and adapt it to your specific machine learning tasks.