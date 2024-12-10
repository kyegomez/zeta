# `VisionEmbedding` Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Purpose and Functionality](#purpose-and-functionality)
3. [Class: `VisionEmbedding`](#class-visionembedding)
   - [Initialization](#initialization)
   - [Parameters](#parameters)
   - [Forward Method](#forward-method)
4. [Usage Examples](#usage-examples)
   - [Using the `VisionEmbedding` Class](#using-the-visionembedding-class)
5. [Additional Information](#additional-information)
   - [Image to Patch Embedding](#image-to-patch-embedding)
6. [References](#references)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the Zeta documentation for the `VisionEmbedding` class! Zeta is a powerful library for deep learning in PyTorch, and this documentation will provide a comprehensive understanding of the `VisionEmbedding` class and its functionalities.

---

## 2. Purpose and Functionality <a name="purpose-and-functionality"></a>

The `VisionEmbedding` class is designed for converting images into patch embeddings, making them suitable for processing by transformer-based models. This class plays a crucial role in various computer vision tasks and enables the integration of vision data into transformer architectures.

---

## 3. Class: `VisionEmbedding` <a name="class-visionembedding"></a>

The `VisionEmbedding` class handles the transformation of images into patch embeddings. It offers flexibility in configuring the embedding process to suit different requirements.

### Initialization <a name="initialization"></a>

To create an instance of the `VisionEmbedding` class, you need to specify the following parameters:

```python
VisionEmbedding(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    contain_mask_token=False,
    prepend_cls_token=False,
)
```

### Parameters <a name="parameters"></a>

- `img_size` (int or tuple, optional): The size of the input image. If a single integer is provided, it is assumed that the image is square. Default is `224`.

- `patch_size` (int or tuple, optional): The size of each patch. If a single integer is provided, square patches are used. Default is `16`.

- `in_chans` (int, optional): The number of input channels in the image. Default is `3` (for RGB images).

- `embed_dim` (int, optional): The dimensionality of the patch embeddings. Default is `768`.

- `contain_mask_token` (bool, optional): Whether to include a mask token in the embeddings. Default is `False`.

- `prepend_cls_token` (bool, optional): Whether to include a class (CLS) token at the beginning of the embeddings. Default is `False`.

### Forward Method <a name="forward-method"></a>

The `forward` method of the `VisionEmbedding` class performs the image-to-patch embedding transformation. It can be called as follows:

```python
output = vision_embedding(input_image, masked_position=None, **kwargs)
```

- `input_image` (Tensor): The input image tensor to be converted into patch embeddings.

- `masked_position` (Tensor, optional): A tensor indicating positions to be masked in the embeddings. This is useful for tasks like image inpainting. Default is `None`.

- `**kwargs`: Additional keyword arguments. These are not mandatory and depend on the specific use case.

---

## 4. Usage Examples <a name="usage-examples"></a>

Let's explore a usage example of the `VisionEmbedding` class to understand how to use it effectively.

### Using the `VisionEmbedding` Class <a name="using-the-visionembedding-class"></a>

```python
import torch

from zeta import VisionEmbedding

# Create an instance of VisionEmbedding
vision_embedding = VisionEmbedding(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    contain_mask_token=True,
    prepend_cls_token=True,
)

# Load an example image (3 channels, 224x224)
input_image = torch.rand(1, 3, 224, 224)

# Perform image-to-patch embedding
output = vision_embedding(input_image)

# The output now contains patch embeddings, ready for input to a transformer model
```

---

## 5. Additional Information <a name="additional-information"></a>

### Image to Patch Embedding <a name="image-to-patch-embedding"></a>

Image to patch embedding is a fundamental step in adapting images for processing by transformer-based models. It divides the image into smaller patches, converts each patch into an embedding, and optionally includes tokens for masking and classification. This process allows transformer models to handle image data effectively.

---

## 6. References <a name="references"></a>

For further information on image to patch embedding and its applications, you can refer to the following resources:

- [Transformers in Computer Vision](https://arxiv.org/abs/2103.08057) - A research paper discussing the application of transformers in computer vision tasks.

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Official PyTorch documentation for related concepts and functions.

This documentation provides a comprehensive overview of the Zeta library's `VisionEmbedding` class and its role in transforming images into patch embeddings. It aims to help you understand the purpose, functionality, and usage of this component for computer vision tasks and transformer-based models.