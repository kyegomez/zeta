# `NaVit` Documentation

## Overview

Welcome to the comprehensive documentation for the Zeta library! Zeta is a sophisticated deep learning framework created to empower machine learning practitioners and researchers with state-of-the-art tools and modules. This documentation will provide an in-depth understanding of the `NaViT` class, its architecture, purpose, parameters, and practical usage.

## Table of Contents

1. [NaViT Class](#navit-class)
   - [Introduction](#introduction)
   - [Architecture](#architecture)
   - [Parameters](#parameters)
   - [Usage Examples](#usage-examples)

## NaViT Class

### Introduction

The `NaViT` class is a vital component of the Zeta library, offering an advanced vision transformer architecture for a wide range of image-related tasks, including image classification. This class incorporates innovative techniques to efficiently process images, handle variable-length sequences, and produce accurate predictions. In this section, we will delve into the architecture, parameters, and usage of the `NaViT` class.

### Architecture

The `NaViT` class follows a complex architecture designed to handle image data effectively:

1. **Image Patching:** The input image is divided into non-overlapping patches, each of which is treated as a separate token. These patches are embedded into a common feature space.

2. **Positional Embeddings:** 2D positional embeddings are added to each patch to encode their spatial information within the image.

3. **Transformer Blocks:** The core of the architecture consists of multiple transformer blocks, which enable capturing global and local features from the patches. These blocks incorporate multi-head self-attention mechanisms and feedforward layers for feature transformation.

4. **Attention Pooling:** At the end of the network, attention pooling is applied to aggregate information from different patches effectively. This enables the model to focus on essential parts of the image.

5. **Output Layer:** The final feature representation is projected onto the output logits, which are used to make predictions.

### Parameters

The `NaViT` class provides various parameters that allow you to customize its behavior. Here are the key parameters:

- `image_size`: The dimensions of the input images (height, width).
- `patch_size`: The size of the image patches used for processing.
- `num_classes`: The number of output classes.
- `dim`: The dimension of the feature embeddings.
- `depth`: The depth of the transformer, indicating the number of transformer blocks.
- `heads`: The number of attention heads in the multi-head self-attention mechanism.
- `mlp_dim`: The dimension of the feedforward neural network within each transformer block.
- `channels`: The number of input image channels (typically 3 for RGB images).
- `dim_head`: The dimension of each attention head.
- `dropout`: Dropout rate applied within the model.
- `emb_dropout`: Dropout rate applied to the patch embeddings.
- `token_dropout_prob`: A parameter that controls token dropout, which can be constant or calculated dynamically based on image dimensions.

### Usage Examples

To illustrate the usage of the `NaViT` class, let's explore three examples:

#### Example 1: Creating a NaViT Model

```python
from zeta import NaViT

# Create a NaViT model with custom parameters
model = NaViT(
    image_size=(256, 256),
    patch_size=32,
    num_classes=1000,
    dim=512,
    depth=12,
    heads=8,
    mlp_dim=1024,
    channels=3,
    dim_head=64,
    dropout=0.1,
    emb_dropout=0.1,
    token_dropout_prob=0.2  # Constant token dropout probability
)
```

In this example, we create a `NaViT` model with specific parameters, including image size, patch size, and network depth.

#### Example 2: Forward Pass with Input Images

```python
import torch

# Generate a batch of input images (list of tensors)
batched_images = [torch.randn(3, 256, 256), torch.randn(3, 128, 128)]

# Perform a forward pass with the NaViT model
preds = model(batched_images)
```

Here, we apply the `NaViT` model to a batch of input images to obtain predictions.

#### Example 3: Extracting Feature Embeddings

```python
# Create a NaViT model with return_embeddings=True to extract feature embeddings
feature_model = NaViT(
    image_size=(256, 256),
    patch_size=32,
    num_classes=1000,
    dim=512,
    depth=12,
    heads=8,
    mlp_dim=1024,
    channels=3,
    dim_head=64,
    dropout=0.1,
    emb_dropout=0.1,
    token_dropout_prob=0.2,
    return_embeddings=True
)

# Forward pass to obtain feature embeddings
feature_embeddings = feature_model(batched_images)
```

In this example, we configure the `NaViT` model to return feature embeddings, allowing you to use the model as a feature extractor.

### Conclusion

This concludes the documentation for the `NaViT` class within the Zeta library. You've gained a comprehensive understanding of its architecture, parameters, and practical usage. The `NaViT` model is a powerful tool for image-related tasks, and with Zeta, you can harness its capabilities to achieve impressive results.

For more information and updates, please refer to the official Zeta documentation and resources.

