# LaplaceActivation


## 1. Overview

The `LaplaceActivation` is an artificial neuron that applies an elementwise activation based on the Laplace function. This was introduced in MEGA as an attention activation, which can be found in this [paper](https://arxiv.org/abs/2209.10655).

The `LaplaceActivation` is inspired by the squaring operation of the ReLU (Rectified Linear Units) function, but comes with a bounded range and gradient for improved stability. 

## 2. Class Description

The `LaplaceActivation` is part of the `PyTorch` neural network (`nn`) module, specifically intended to provide activation functionality based on the Laplace function to a neural network model. 

### Class Definition

```python
class LaplaceActivation(nn.Module):
    pass
```

### Method: `forward`

This function applies the Laplace function across all elements in the input tensor. It takes as parameters the input tensor and optional parameters `\mu` and `\sigma`.
The function computes the Laplace function as follows:

```
input = (input - \mu) / (\sigma * sqrt(2))
output = 0.5 * (1 + erf(input))
return output
```
#### Arguments:

|Argument|Type |Description |Default value
|---|---|---|---|
|`input` |Tensor| Tensor input to the function.|
|`\mu` |float|Location parameter, `\mu` determines the shift or the mean of the function.|0.707107
|`\sigma`|float| Scale parameter or standard deviation, `\sigma` determines the spread or the width of the function.| 0.282095

#### Returns

A tensor with Laplace function applied elementwise.

### 3. Example Usage

#### Importing required libraries

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from zeta.nn import LaplaceActivation
```
#### Defining an instance

```python
lap_act = LaplaceActivation()
```
Applying Laplace Activation to a tensor

```python
input_tensor = torch.randn(10)
activated_tensor = lap_act(input_tensor)
```
Printing output

```python
print(activated_tensor)
```

You should see the tensor output with Laplace activation applied elementwise.

## 4. Additional Information

The Laplace Activation function is a new approach to help stabilize the learning process in deep neural networks. It introduces bounded range and gradient which can be very useful when training deep learning models.

## 5. References 

For more in-depth understanding, kindly refer to this [paper](https://arxiv.org/abs/2209.10655).

## 6. Contact Information

For any issues or inquiries, feel free to contact the support team at kye@apac.ai We're happy to help!

