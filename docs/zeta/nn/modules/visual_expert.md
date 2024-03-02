# `VisualExpert` Module Documentation

**Table of Contents**

- [Introduction](#introduction)
- [Module Overview](#module-overview)
- [Class Definition](#class-definition)
  - [Parameters](#parameters)
- [Functionality and Usage](#functionality-and-usage)
  - [How Visual Expert Works](#how-visual-expert-works)
  - [Usage Examples](#usage-examples)
- [Additional Information and Tips](#additional-information-and-tips)
- [References](#references)

## Introduction <a name="introduction"></a>

Welcome to the documentation for the Visual Expert module, a component inspired by the research paper [Visual Expert module](https://arxiv.org/pdf/2311.03079.pdf). This module is designed to enable deep visual-language feature alignment, making it a valuable addition to your deep learning projects involving both text and image data. In this comprehensive guide, we will explore the purpose, functionality, and usage of the Visual Expert module.

## Module Overview <a name="module-overview"></a>

The Visual Expert module is a crucial component for enhancing deep visual-language feature alignment. It consists of a QKV (Query, Key, Value) matrix and a Multi-Layer Perceptron (MLP) in each layer. These components have the same shapes as those in pretrained language models and are initialized from them. The primary motivation behind the Visual Expert module is to align image features with the different attention heads in a language model, enabling deep fusion.

## Class Definition <a name="class-definition"></a>

The VisualExpert class in this module encapsulates the functionality needed to perform deep visual-language feature alignment. Let's explore its parameters and how to use it effectively.

```python
class VisualExpert:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
        heads: int,
    ):
        ...

    def __call__(self, x: torch.Tensor):
        ...
```

### Parameters <a name="parameters"></a>

| Parameter     | Type   | Description                                           |
|---------------|--------|-------------------------------------------------------|
| `dim`         | int    | The dimension of the input features.                 |
| `hidden_dim`  | int    | The dimension of the hidden layer in the feedforward.|
| `dropout`     | float  | The dropout rate.                                    |
| `heads`       | int    | The number of heads in the multihead attention.      |

## Functionality and Usage <a name="functionality-and-usage"></a>

### How Visual Expert Works <a name="how-visual-expert-works"></a>

The Visual Expert module works by aligning image features with different attention heads in a language model. Here's a step-by-step explanation of how it operates:

1. The input hidden states of an attention layer are represented as `X`, where:
   - `X` has shape `B×H×(LI+LT)×D`.
   - `B` is the batch size.
   - `LI` and `LT` are the lengths of image and text sequences.
   - `H` is the number of attention heads.
   - `D` is the hidden size.

2. In the attention with the Visual Expert, `X` is initially split into text and image features.

3. QKV projections are applied separately for text and image features:
   - Query (`q_text`, `q_img`)
   - Key (`k_text`, `k_img`)
   - Value (`v_text`, `v_img`)

4. Attention is applied with the image features appended in front of the text features. The `q`, `k`, and `v` of text and images are concatenated together.

5. The attention output is added to the normalized input (`X`) to capture feature alignment.

6. Another layer normalization is applied.

7. Text and image features are separated.

8. Feedforward layers are applied to both text and image features.

9. The output of the feedforwards is added together with the output of the added attention and normalization.

### Usage Examples <a name="usage-examples"></a>

#### Example 1: Creating a Visual Expert Module

```python
import torch

from zeta.nn import VisualExpert

# Create a Visual Expert module
visual_expert = VisualExpert(dim=1024, hidden_dim=2048, dropout=0.1, heads=16)
```

#### Example 2: Forward Pass

```python
# Generate a random input tensor
x = torch.randn(1, 10, 1024)

# Apply the Visual Expert module
output = visual_expert(x)

# Check the output shape
print(output.shape)  # torch.Size([1, 10, 1024])
```

#### Example 3: Customizing Visual Expert

You can customize the Visual Expert module by adjusting its parameters.

```python
# Create a Visual Expert module with different parameters
visual_expert_custom = VisualExpert(dim=512, hidden_dim=1024, dropout=0.2, heads=8)

# Apply it to your data
output_custom = visual_expert_custom(x)
```

## Additional Information and Tips <a name="additional-information-and-tips"></a>

- Experiment with different values for the `dim`, `hidden_dim`, `dropout`, and `heads` parameters to fine-tune the Visual Expert module for your specific tasks.

- Ensure that your input data shapes match the expected shapes described in the module documentation.

- If working with image and text data, preprocess and format your data accordingly before applying the Visual Expert module.

- Keep in mind that this module is designed for deep visual-language feature alignment, making it suitable for tasks that involve both text and image data.

## References <a name="references"></a>

- Research Paper: [Visual Expert module](https://arxiv.org/pdf/2311.03079.pdf)

- PyTorch Documentation: [PyTorch](https://pytorch.org/docs/stable/index.html)

This concludes the documentation for the Visual Expert module. We hope this guide helps you understand its purpose, functionality, and how to use it effectively in your deep learning projects.