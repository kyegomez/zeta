# `TimeUpSample2x` Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Overview](#overview)
3. [TimeUpSample2x Class](#timeupsample2x-class)
   - [Initialization Parameters](#initialization-parameters)
4. [Functionality and Usage](#functionality-and-usage)
   - [Forward Method](#forward-method)
5. [Utility Functions](#utility-functions)
6. [Examples](#examples)
   - [Example 1: Creating a TimeUpSample2x Module](#example-1-creating-a-timeupsample2x-module)
   - [Example 2: Using TimeUpSample2x for Upsampling](#example-2-using-timeupsample2x-for-upsampling)
7. [Additional Information](#additional-information)
8. [References and Resources](#references-and-resources)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the documentation for the Zeta library. This comprehensive guide provides detailed information about the Zeta library and its components, focusing on the `TimeUpSample2x` class. Before we delve into the details, it's important to understand the purpose and significance of this library.

### 1.1 Purpose

The Zeta library is designed to simplify the development of deep learning models by offering modular components and utilities. One of these components is the `TimeUpSample2x` class, which plays a crucial role in upscaling the time dimension of tensors.

### 1.2 Key Features

- **Time Dimension Upsampling:** The `TimeUpSample2x` class allows you to efficiently increase the temporal resolution of your data, which is particularly valuable in various sequential data tasks.

- **Seamless Integration:** Zeta modules seamlessly integrate with popular deep learning frameworks like PyTorch, making it easy to incorporate them into your projects.

---

## 2. Overview <a name="overview"></a>

The Zeta library is built with the aim of providing essential building blocks for deep learning model development. One such block is the `TimeUpSample2x` class.

### 2.1 `TimeUpSample2x` Class

The `TimeUpSample2x` class is a module designed for upscaling the time dimension of 3D tensors. It is useful in scenarios where increasing the temporal resolution of the data is required.

In the following sections, we will explore the `TimeUpSample2x` class's definition, initialization parameters, functionality, and usage.

---

## 3. TimeUpSample2x Class <a name="timeupsample2x-class"></a>

The `TimeUpSample2x` class is at the core of Zeta, providing the ability to increase the temporal resolution of tensors.

### 3.1 Initialization Parameters <a name="initialization-parameters"></a>

Here are the initialization parameters for the `TimeUpSample2x` class:

- `dim` (int): The number of input channels in the tensor.

- `dim_out` (int, optional): The number of output channels in the tensor after upsampling. If not specified, it defaults to the same as `dim`.

### 3.2 Methods

The primary method of the `TimeUpSample2x` class is the `forward` method, which performs the time dimension upsampling operation on input tensors.

---

## 4. Functionality and Usage <a name="functionality-and-usage"></a>

Let's explore the functionality and usage of the `TimeUpSample2x` class.

### 4.1 Forward Method <a name="forward-method"></a>

The `forward` method of the `TimeUpSample2x` class takes an input tensor and applies time dimension upsampling using a convolution operation. Here is the parameter:

- `x` (Tensor): The input tensor of shape `(batch, channels, time, height, width)`.

The method returns an upsampled tensor of shape `(batch, output_channels, time, height, width)`.

### 4.2 Usage Examples <a name="usage-examples"></a>

#### Example 1: Creating a TimeUpSample2x Module <a name="example-1-creating-a-timeupsample2x-module"></a>

In this example, we create an instance of the `TimeUpSample2x` class with default settings:

```python
upsample = TimeUpSample2x(dim=64)
```

#### Example 2: Using TimeUpSample2x for Upsampling <a name="example-2-using-timeupsample2x-for-upsampling"></a>

Here, we demonstrate how to use the `TimeUpSample2x` module for upsampling an input tensor:

```python
upsample = TimeUpSample2x(dim=64)
input_data = torch.randn(1, 64, 32, 32)
output = upsample(input_data)
print(output.shape)
```

---

## 5. Utility Functions <a name="utility-functions"></a>

The Zeta library also provides a set of utility functions used within the modules. These utility functions, such as `exists`, `identity`, `divisible_by`, and more, enhance the modularity and flexibility of the library.

---

## 6. Additional Information <a name="additional-information"></a>

Here are some additional tips and information for using the Zeta library and the `TimeUpSample2x` class effectively:

- Experiment with different values for the `dim` and `dim_out` parameters to control the number of channels in the output tensor.

- Ensure that the input tensor (`x`) has the appropriate shape `(batch, channels, time, height, width)`.

---

## 7. References and Resources <a name="references-and-resources"></a>

For further information and resources related to the Zeta library and deep learning, please refer to the following:

- [Zeta GitHub Repository](https://github.com/kyegomez/zeta): The official Zeta repository for updates and contributions.

- [PyTorch Official Website](https://pytorch.org/): The official website for PyTorch, the deep learning framework used in Zeta.

This concludes the documentation for the Zeta library and the `TimeUpSample2x` class. You now have a comprehensive understanding of how to use this library and module for your deep learning projects. If you have any further questions or need assistance, please refer to the provided references and resources. Happy modeling with Zeta!