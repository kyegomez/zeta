# `SpatialDownsample` Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Overview](#overview)
3. [SpatialDownsample Class](#spatialdownsample-class)
   - [Initialization Parameters](#initialization-parameters)
4. [Functionality and Usage](#functionality-and-usage)
   - [Forward Method](#forward-method)
5. [Utility Functions](#utility-functions)
6. [Examples](#examples)
   - [Example 1: Creating a SpatialDownsample Module](#example-1-creating-a-spatialdownsample-module)
   - [Example 2: Using SpatialDownsample for Downsampling](#example-2-using-spatialdownsample-for-downsampling)
7. [Additional Information](#additional-information)
8. [References and Resources](#references-and-resources)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the documentation for the Zeta library. This documentation will provide you with comprehensive information on the Zeta library, specifically focusing on the `SpatialDownsample` class. Before we dive into the details, let's understand the purpose and significance of this library.

### 1.1 Purpose

The Zeta library is designed to provide essential building blocks for deep learning architectures, making it easier for researchers and developers to implement complex models. It offers various modules and utilities, including the `SpatialDownsample` class, which is a key component for downsampling spatial dimensions in neural networks.

### 1.2 Key Features

- **Spatial Downsampling:** The `SpatialDownsample` class allows you to efficiently reduce the spatial dimensions of your data, which is crucial for various computer vision tasks.

- **Integration:** Zeta modules seamlessly integrate with popular deep learning frameworks like PyTorch, enabling you to incorporate them into your projects effortlessly.

---

## 2. Overview <a name="overview"></a>

The Zeta library aims to simplify deep learning model development by providing modular components that adhere to best practices in the field. One such component is the `SpatialDownsample` class.

### 2.1 `SpatialDownsample` Class

The `SpatialDownsample` class is a module designed for spatial downsampling of 3D tensors. It plays a critical role in architectures like ResNet, where downsampling is necessary to reduce spatial dimensions while increasing the number of channels.

In the following sections, we will explore the `SpatialDownsample` class's definition, initialization parameters, functionality, and usage.

---

## 3. SpatialDownsample Class <a name="spatialdownsample-class"></a>

The `SpatialDownsample` class is at the core of Zeta, providing spatial downsampling capabilities for 3D tensors.

### 3.1 Initialization Parameters <a name="initialization-parameters"></a>

Here are the initialization parameters for the `SpatialDownsample` class:

- `dim` (int): The number of input channels in the tensor.

- `dim_out` (int, optional): The number of output channels in the tensor after downsampling. If not specified, it defaults to the same as `dim`.

- `kernel_size` (int): The size of the kernel used for downsampling. It determines the amount of spatial reduction in the output tensor.

### 3.2 Methods

The primary method of the `SpatialDownsample` class is the `forward` method, which performs the spatial downsampling operation on input tensors.

---

## 4. Functionality and Usage <a name="functionality-and-usage"></a>

Let's delve into the functionality and usage of the `SpatialDownsample` class.

### 4.1 Forward Method <a name="forward-method"></a>

The `forward` method of the `SpatialDownsample` class takes an input tensor and applies spatial downsampling using a convolution operation. Here are the parameters:

- `x` (Tensor): The input tensor of shape `(batch, channels, time, height, width)`.

The method returns a downsampled tensor of shape `(batch, output_channels, time, height, width)`.

### 4.2 Usage Examples <a name="usage-examples"></a>

#### Example 1: Creating a SpatialDownsample Module <a name="example-1-creating-a-spatialdownsample-module"></a>

In this example, we create an instance of the `SpatialDownsample` class with default settings:

```python
downsample = SpatialDownsample(dim=64, kernel_size=3)
```

#### Example 2: Using SpatialDownsample for Downsampling <a name="example-2-using-spatialdownsample-for-downsampling"></a>

Here, we demonstrate how to use the `SpatialDownsample` module for downsampling an input tensor:

```python
downsample = SpatialDownsample(dim=64, kernel_size=3)
input_data = torch.randn(1, 64, 32, 32)
output = downsample(input_data)
print(output.shape)
```

---

## 5. Utility Functions <a name="utility-functions"></a>

The Zeta library also provides a set of utility functions used within the modules. These utility functions, such as `exists`, `default`, `identity`, and more, contribute to the modularity and flexibility of the library.

---

## 6. Additional Information <a name="additional-information"></a>

Here are some additional tips and information for using the Zeta library and the `SpatialDownsample` class effectively:

- Experiment with different kernel sizes to control the amount of downsampling according to your specific model requirements.

- Ensure that the input tensor (`x`) has the appropriate shape `(batch, channels, time, height, width)`.

---

## 7. References and Resources <a name="references-and-resources"></a>

For further information and resources related to the Zeta library and deep learning, please refer to the following:

- [Zeta GitHub Repository](https://github.com/kyegomez/zeta): The official Zeta repository for updates and contributions.

- [ResNet Paper](https://arxiv.org/abs/1512.03385): The original ResNet paper that introduces the concept of spatial downsampling.

- [PyTorch Official Website](https://pytorch.org/): The official website for PyTorch, the deep learning framework used in Zeta.

This concludes the documentation for the Zeta library and the `SpatialDownsample` class. You now have a comprehensive understanding of how to use this library and module for your deep learning projects. If you have any further questions or need assistance, please refer to the provided references and resources. Happy modeling with Zeta!