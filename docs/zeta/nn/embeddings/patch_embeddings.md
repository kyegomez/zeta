# `PatchEmbeddings` Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Purpose and Functionality](#purpose-and-functionality)
3. [Class: `PatchEmbeddings`](#class-patchembeddings)
   - [Parameters](#parameters)
4. [Usage Examples](#usage-examples)
   - [Using the `PatchEmbeddings` Class](#using-the-patchembeddings-class)
5. [Additional Information](#additional-information)
6. [References](#references)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the Zeta documentation! In this documentation, we will explore the `PatchEmbeddings` class, which is part of the Zeta library. This class plays a crucial role in the field of computer vision, particularly in the context of image processing and deep learning. This documentation aims to provide a comprehensive understanding of its purpose, functionality, and usage.

---

## 2. Purpose and Functionality <a name="purpose-and-functionality"></a>

The `PatchEmbeddings` class serves as a fundamental component for processing images in deep learning models, such as transformers and convolutional neural networks (CNNs). Its primary functionalities include:

### Patch Embedding

- **Image Segmentation**: It segments an input image into smaller patches, which are then individually processed by the neural network.

- **Dimensionality Transformation**: It transforms the dimensionality of each patch, preparing them for further processing.

- **Normalization**: It applies layer normalization to the patch embeddings for improved training stability.

### Image-to-Sequence Transformation

- **Reshaping**: The class reshapes the image patches into a sequence of vectors suitable for input to models like transformers.

- **Linear Projection**: It uses a linear layer to project the patch embeddings into the desired output dimension.

### Versatility

- **Configurability**: You can configure the input and output dimensions, allowing flexibility for various model architectures.

- **Normalization Control**: It provides control over the layer normalization applied to the embeddings.

---

## 3. Class: `PatchEmbeddings` <a name="class-patchembeddings"></a>

The `PatchEmbeddings` class has the following signature:

```python
class PatchEmbeddings(nn.Module):
    def __init__(
        self, 
        dim_in, 
        dim_out, 
        seq_len
    )
    
    def forward(self, x)
```

### Parameters <a name="parameters"></a>

- `dim_in` (int): The input dimension of the image patches.

- `dim_out` (int): The output dimension after embedding the patches.

- `seq_len` (int): The length of the sequence after patching the image.

---

## 4. Usage Examples <a name="usage-examples"></a>

Let's explore how to use the `PatchEmbeddings` class effectively in various scenarios.

### Using the `PatchEmbeddings` Class <a name="using-the-patchembeddings-class"></a>

Here's how to use the `PatchEmbeddings` class to embed image patches:

```python
import torch
from zeta.vision import PatchEmbeddings

# Define the input image properties
dim_in = 3  # Input dimension of image patches (e.g., 3 for RGB images)
dim_out = 64  # Output dimension after embedding
seq_len = 16  # Length of the sequence after patching the image

# Create an instance of PatchEmbeddings
patch_embed = PatchEmbeddings(dim_in, dim_out, seq_len)

# Create a random input image tensor (batch_size, channels, height, width)
image = torch.randn(32, dim_in, 224, 224)  # Example input image with 32 samples

# Apply patch embedding
embedded_patches = patch_embed(image)

# Print the embedded patches
print(embedded_patches)
```

---

## 5. Additional Information <a name="additional-information"></a>

Here are some additional notes and tips related to the `PatchEmbeddings` class:

- **Image Patching**: Patching images is a common technique used to process large images in deep learning models.

- **Normalization**: The application of layer normalization helps stabilize training and improve convergence.

- **Dimensionality Transformation**: Patch embeddings are essential for converting spatial information in images into sequences suitable for neural networks.

- **Versatile Usage**: The `PatchEmbeddings` class can be used in various vision-based deep learning architectures.

---

## 6. References <a name="references"></a>

For further information on image patching, layer normalization, and related concepts, you can refer to the following resources:

- [Vision Transformers (ViT) - Paper](https://arxiv.org/abs/2010.11929) - The original research paper introducing Vision Transformers.

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Official PyTorch documentation for related functions and modules.

This documentation provides a comprehensive overview of the Zeta library's `PatchEmbeddings` class. It aims to help you understand the purpose, functionality, and usage of this class for image patch embedding, which is a crucial step in various computer vision applications and deep learning models.