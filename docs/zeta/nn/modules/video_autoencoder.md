# `CausalConv3d` Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Overview](#overview)
3. [CausalConv3d Class](#causalconv3d-class)
   - [Initialization Parameters](#initialization-parameters)
4. [Functionality and Usage](#functionality-and-usage)
   - [Forward Method](#forward-method)
5. [Utility Functions](#utility-functions)
6. [Examples](#examples)
   - [Example 1: Creating a CausalConv3d Module](#example-1-creating-a-causalconv3d-module)
   - [Example 2: Using CausalConv3d for Causal Convolution](#example-2-using-causalconv3d-for-causal-convolution)
7. [Additional Information](#additional-information)
8. [References and Resources](#references-and-resources)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the documentation for the Zeta library. This comprehensive guide provides detailed information about the Zeta library and its components, focusing on the `CausalConv3d` class. Before we delve into the details, it's important to understand the purpose and significance of this library.

### 1.1 Purpose

The Zeta library is designed to simplify the development of deep learning models by offering modular components and utilities. One of these components is the `CausalConv3d` class, which plays a crucial role in performing causal convolutions on 3D tensors.

### 1.2 Key Features

- **Causal Convolution:** The `CausalConv3d` class enables causal convolutions on 3D tensors, a vital operation in architectures like ResNet.

- **Seamless Integration:** Zeta modules seamlessly integrate with popular deep learning frameworks like PyTorch, making it easy to incorporate them into your projects.

---

## 2. Overview <a name="overview"></a>

The Zeta library is built with the aim of providing essential building blocks for deep learning model development. One such block is the `CausalConv3d` class.

### 2.1 `CausalConv3d` Class

The `CausalConv3d` class is a module designed for performing causal convolutions on 3D tensors. It is particularly useful in scenarios where preserving the causality of data is essential, such as in ResNet architectures.

In the following sections, we will explore the `CausalConv3d` class's definition, initialization parameters, functionality, and usage.

---

## 3. CausalConv3d Class <a name="causalconv3d-class"></a>

The `CausalConv3d` class is at the core of Zeta, providing the ability to perform causal convolutions on 3D tensors.

### 3.1 Initialization Parameters <a name="initialization-parameters"></a>

Here are the initialization parameters for the `CausalConv3d` class:

- `chan_in` (int): The number of input channels in the tensor.

- `chan_out` (int): The number of output channels in the tensor after convolution.

- `kernel_size` (int or Tuple[int, int, int]): The size of the convolution kernel. It can be a single integer or a tuple specifying the size in three dimensions.

- `pad_mode` (str): The padding mode used for the convolution operation.

- `**kwargs` (dict): Additional arguments to be passed to the convolution layer.

### 3.2 Methods

The primary method of the `CausalConv3d` class is the `forward` method, which performs the causal convolution operation on input tensors.

---

## 4. Functionality and Usage <a name="functionality-and-usage"></a>

Let's explore the functionality and usage of the `CausalConv3d` class.

### 4.1 Forward Method <a name="forward-method"></a>

The `forward` method of the `CausalConv3d` class takes an input tensor and applies causal convolution using a 3D convolutional layer. Here is the parameter:

- `x` (Tensor): The input tensor of shape `(batch, channels, time, height, width)`.

The method returns a tensor after performing causal convolution, preserving causality in the temporal dimension.

### 4.2 Usage Examples <a name="usage-examples"></a>

#### Example 1: Creating a CausalConv3d Module <a name="example-1-creating-a-causalconv3d-module"></a>

In this example, we create an instance of the `CausalConv3d` class with default settings:

```python
causal_conv = CausalConv3d(chan_in=64, chan_out=128, kernel_size=3)
```

#### Example 2: Using CausalConv3d for Causal Convolution <a name="example-2-using-causalconv3d-for-causal-convolution"></a>

Here, we demonstrate how to use the `CausalConv3d` module for performing causal convolution on an input tensor:

```python
causal_conv = CausalConv3d(chan_in=64, chan_out=128, kernel_size=3)
input_data = torch.randn(1, 64, 32, 32)
output = causal_conv(input_data)
print(output.shape)
```

---

## 5. Utility Functions <a name="utility-functions"></a>

The Zeta library also provides a set of utility functions used within the modules. These utility functions, such as `exists`, `default`, `identity`, and more, enhance the modularity and flexibility of the library.

---

## 6. Additional Information <a name="additional-information"></a>

Here are some additional tips and information for using the Zeta library and the `CausalConv3d` class effectively:

- Experiment with different values for `chan_in`, `chan_out`, and `kernel_size` to control the number of input and output channels and the convolution kernel size.

- Ensure that the input tensor (`x`) has the appropriate shape `(batch, channels, time, height, width)` to perform causal convolution.

- The `pad_mode` parameter allows you to specify the padding mode for the convolution operation.

---

## 7. References and Resources <a name="references-and-resources"></a>

For further information and resources related to the Zeta library and deep learning, please refer to the following:

- [Zeta GitHub Repository](https://github.com/Zeta): The official Zeta repository for updates and contributions.

- [PyTorch Official Website](https://pytorch.org/): The official website for PyTorch, the deep learning framework used in Zeta.

This concludes the documentation for the Zeta library and the `CausalConv3d` class. You now have a comprehensive understanding of how to use this library and module for your deep learning projects. If you have any further questions or need assistance, please refer to the provided references and resources. Happy modeling with Zeta!