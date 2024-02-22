# `MaxVit` Documentation

## Overview

Welcome to the documentation for the Zeta library! Zeta is a state-of-the-art deep learning framework designed to empower researchers and practitioners in the field of machine learning. With a focus on modularity, efficiency, and performance, Zeta offers a wide range of tools and modules for building and training advanced neural network models.

This comprehensive documentation will provide a deep and thorough understanding of the code for the `MaxVit` class within the Zeta library. We will cover its architecture, purpose, parameters, and usage through extensive examples. By the end of this documentation, you'll have a solid grasp of how to leverage the power of `MaxVit` in your machine learning projects.

## Table of Contents

1. [MaxVit Class](#maxvit-class)
   - [Introduction](#introduction)
   - [Architecture](#architecture)
   - [Parameters](#parameters)
   - [Usage Examples](#usage-examples)

## MaxVit Class

### Introduction

The `MaxVit` class is a key component of the Zeta library, offering a powerful vision transformer architecture for image classification tasks. It incorporates cutting-edge techniques to efficiently process image data and produce accurate predictions. In this section, we will dive deep into the architecture, parameters, and usage of the `MaxVit` class.

### Architecture

The `MaxVit` class follows a multi-stage architecture to process image data effectively:

1. **Convolutional Stem:** The input image undergoes a series of convolutional layers to extract basic features. This step serves as the initial feature extraction.

2. **Transformer Stages:** `MaxVit` consists of multiple transformer stages, each containing several transformer blocks. These stages allow the model to capture hierarchical features at different resolutions. Each transformer block includes the following components:
   - MBConv: MobileNetV2-like convolutional block with expansion and shrinkage.
   - Grid Attention: Efficient attention mechanism tailored for grid-like structures.
   - FeedForward: Standard feedforward neural network layer.
   
3. **MLP Head:** The final feature representation is passed through a multi-layer perceptron (MLP) head to produce predictions.

### Parameters

The `MaxVit` class accepts various parameters that allow you to customize its behavior. Here are the key parameters:

- `num_classes`: The number of output classes.
- `dim`: The dimension of the feature embeddings.
- `depth`: A tuple indicating the number of transformer blocks at each stage.
- `dim_head`: The dimension of each attention head.
- `dim_conv_stem`: The dimension of the convolutional stem output.
- `window_size`: The size of the window for efficient grid-like attention.
- `mbconv_expansion_rate`: The expansion rate for the MBConv blocks.
- `mbconv_shrinkage_rate`: The shrinkage rate for the MBConv blocks.
- `dropout`: Dropout rate applied within the model.
- `channels`: The number of input image channels.

### Usage Examples

To illustrate the usage of the `MaxVit` class, let's explore three examples:

#### Example 1: Creating a MaxVit Model

```python
from zeta import MaxVit

# Create a MaxVit model with custom parameters
model = MaxVit(
    num_classes=1000,
    dim=512,
    depth=(2, 2, 6, 3),  # Depth of transformer blocks at each stage
    dim_head=32,
    dim_conv_stem=64,
    window_size=7,
    mbconv_expansion_rate=4,
    mbconv_shrinkage_rate=0.25,
    dropout=0.01,
    channels=3,
)
```

In this example, we create a `MaxVit` model with specific parameters, including the number of classes and model depth.

#### Example 2: Forward Pass with Input Image

```python
import torch

# Generate a random input image
img = torch.randn(1, 3, 256, 256)

# Perform a forward pass with the MaxVit model
preds = model(img)
```

Here, we apply the `MaxVit` model to a random input image to obtain predictions.

#### Example 3: Extracting Feature Embeddings

```python
# Create a MaxVit model with return_embeddings=True to extract feature embeddings
feature_model = MaxVit(
    num_classes=1000,
    dim=512,
    depth=(2, 2, 6, 3),
    dim_head=32,
    dim_conv_stem=64,
    window_size=7,
    mbconv_expansion_rate=4,
    mbconv_shrinkage_rate=0.25,
    dropout=0.01,
    channels=3
    return_embeddings=True
)

# Forward pass to obtain feature embeddings
feature_embeddings = feature_model(img)
```

In this example, we configure the `MaxVit` model to return feature embeddings, allowing you to use the model as a feature extractor.

### Conclusion

This concludes the documentation for the `MaxVit` class within the Zeta library. You've gained a comprehensive understanding of its architecture, parameters, and practical usage. The `MaxVit` model is a powerful tool for image classification tasks, and with Zeta, you can leverage its capabilities to achieve impressive results.

For more information and updates, please refer to the official Zeta documentation and resources.
