# `AdaptiveConv3DMod` Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Overview](#overview)
3. [AdaptiveConv3DMod Class](#adaptiveconv3dmod-class)
   - [Initialization Parameters](#initialization-parameters)
4. [Functionality and Usage](#functionality-and-usage)
   - [Forward Method](#forward-method)
5. [Helper Functions and Classes](#helper-functions-and-classes)
6. [Examples](#examples)
   - [Example 1: Creating an AdaptiveConv3DMod Layer](#example-1-creating-an-adaptiveconv3dmod-layer)
   - [Example 2: Using AdaptiveConv3DMod with Modulation](#example-2-using-adaptiveconv3dmod-with-modulation)
7. [Additional Information](#additional-information)
8. [References and Resources](#references-and-resources)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the documentation for the Zeta library's `AdaptiveConv3DMod` class. This class implements an adaptive convolutional layer with support for spatial modulation, as used in the StyleGAN2 architecture. This documentation will provide you with a comprehensive understanding of how to use the `AdaptiveConv3DMod` class for various tasks.

### 1.1 Purpose

The primary purpose of the `AdaptiveConv3DMod` class is to enable adaptive convolutional operations with optional spatial modulation. It is particularly useful in tasks that involve conditional generation, where the convolutional layer's weights are modulated based on external factors or latent variables.

### 1.2 Key Features

- Adaptive convolutional layer for 3D data.
- Support for spatial modulation to condition the convolution.
- Demodulation option for weight normalization.
- Flexible and customizable for various architectural designs.

---

## 2. Overview <a name="overview"></a>

Before diving into the details of the `AdaptiveConv3DMod` class, let's provide an overview of its purpose and functionality.

The `AdaptiveConv3DMod` class is designed to perform convolutional operations on 3D data while allowing for dynamic modulation of the convolutional weights. This modulation is particularly useful in generative models where conditional generation is required. The class provides options for demodulation and flexible kernel sizes.

In the following sections, we will explore the class definition, its initialization parameters, and how to use it effectively.

---

## 3. AdaptiveConv3DMod Class <a name="adaptiveconv3dmod-class"></a>

The `AdaptiveConv3DMod` class is the core component of the Zeta library for adaptive convolutional operations. It provides methods for performing convolution with optional spatial modulation.

### 3.1 Initialization Parameters <a name="initialization-parameters"></a>

Here are the initialization parameters for the `AdaptiveConv3DMod` class:

- `dim` (int): The number of input channels, i.e., the dimension of the input data.

- `spatial_kernel` (int): The size of the spatial kernel used for convolution.

- `time_kernel` (int): The size of the temporal (time) kernel used for convolution.

- `dim_out` (int, optional): The number of output channels, which can be different from the input dimension. If not specified, it defaults to the input dimension.

- `demod` (bool): If `True`, demodulates the weights during convolution to ensure proper weight normalization.

- `eps` (float): A small value added for numerical stability to prevent division by zero.

### 3.2 Attributes

The `AdaptiveConv3DMod` class has the following important attributes:

- `weights` (nn.Parameter): The learnable convolutional weights.

- `padding` (tuple): The padding configuration for the convolution operation based on the kernel size.

### 3.3 Methods

The main method of the `AdaptiveConv3DMod` class is the `forward` method, which performs the forward pass of the convolution operation with optional modulation.

---

## 4. Functionality and Usage <a name="functionality-and-usage"></a>

Now let's explore how to use the `AdaptiveConv3DMod` class for convolution operations with optional modulation.

### 4.1 Forward Method <a name="forward-method"></a>

The `forward` method is used to perform the forward pass of the adaptive convolutional layer. It takes the following parameters:

- `fmap` (Tensor): The input feature map or data of shape `(batch, channels, time, height, width)`.

- `mod` (Optional[Tensor]): An optional modulation tensor that conditions the convolutional weights. It should have the shape `(batch, channels)`.

The method returns a tensor of shape `(batch, output_channels, time, height, width)`.

Example:

```python
layer = AdaptiveConv3DMod(dim=512, spatial_kernel=3, time_kernel=3)
input_data = torch.randn(1, 512, 4, 4, 4)
modulation = torch.randn(1, 512)
output = layer(input_data, modulation)
print(output.shape)
```

### 4.2 Usage Examples <a name="usage-examples"></a>

#### Example 1: Creating an AdaptiveConv3DMod Layer <a name="example-1-creating-an-adaptiveconv3dmod-layer"></a>

In this example, we create an instance of the `AdaptiveConv3DMod` class with default settings:

```python
layer = AdaptiveConv3DMod(dim=512, spatial_kernel=3, time_kernel=3)
```

#### Example 2: Using AdaptiveConv3DMod with Modulation <a name="example-2-using-adaptiveconv3dmod-with-modulation"></a>

Here, we demonstrate how to use the `AdaptiveConv3DMod` layer with modulation:

```python
layer = AdaptiveConv3DMod(dim=512, spatial_kernel=3, time_kernel=3)
input_data = torch.randn(1, 512, 4, 4, 4)
modulation = torch.randn(1, 512)
output = layer(input_data, modulation)
print(output.shape)
```

---

## 5. Helper Functions and Classes <a name="helper-functions-and-classes"></a>

The Zeta library provides several helper functions and classes that are used within the `AdaptiveConv3DMod` class. These include functions for checking divisibility, packing and unpacking tensors, and more. These helper functions contribute to the functionality and flexibility of the `AdaptiveConv3DMod` class.

---

## 6. Additional Information <a name="additional-information"></a>

Here are some additional tips and information for using the `AdaptiveConv3DMod` class effectively:

- Experiment with different spatial and temporal kernel sizes to match the requirements of your specific task.

- Be cautious when enabling demodulation, as it may affect the convergence of the model. You can adjust the `eps` parameter for better stability.

- Ensure that your modulation tensor (`mod`) has the appropriate shape and values to condition the convolutional weights effectively.

---

## 7. References and Resources <a name="references-and-resources"></a>

Here are some references and resources for further information on the Zeta library and related topics:



- [Zeta GitHub Repository](https://github.com/kyegomez/zeta): Official Zeta repository for updates and contributions.

- [StyleGAN2 Paper](https://arxiv.org/abs/1912.04958): The original paper that introduces adaptive convolution with modulation.

- [PyTorch Official Website](https://pytorch.org/): Official website for PyTorch, the deep learning framework used in Zeta.

This concludes the documentation for the Zeta library's `AdaptiveConv3DMod` class. You now have a comprehensive understanding of how to use this class for adaptive convolution operations with modulation. If you have any further questions or need assistance, please refer to the provided references and resources. Happy modeling with Zeta!