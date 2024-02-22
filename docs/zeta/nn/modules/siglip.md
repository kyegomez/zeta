# SigLipLoss Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Overview](#overview)
3. [Installation](#installation)
4. [Usage](#usage)
   1. [Initializing SigLipLoss](#initializing-sigliploss)
   2. [Calculating Loss](#calculating-loss)
   3. [Multi-process Communication](#multi-process-communication)
5. [Examples](#examples)
   1. [Example 1: Initializing SigLipLoss](#example-1-initializing-sigliploss)
   2. [Example 2: Calculating Loss](#example-2-calculating-loss)
   3. [Example 3: Multi-process Communication](#example-3-multi-process-communication)
6. [Additional Information](#additional-information)
7. [Conclusion](#conclusion)

---

## 1. Introduction <a name="introduction"></a>

The `SigLipLoss` module is a component of the **SigLIP (Sigmoid Loss for Language Image Pre-Training)** framework, designed to facilitate efficient training of models for language-image pre-training tasks. SigLIP is particularly useful for scenarios where you need to pre-train a model to understand the relationship between text and images.

This documentation provides a comprehensive guide to using the `SigLipLoss` module, including its purpose, parameters, and usage examples.

---

## 2. Overview <a name="overview"></a>

The `SigLipLoss` module is used to compute the loss for training models in the SigLIP framework. It calculates the contrastive loss between image and text features, which is a fundamental component of the SigLIP training process.

Key features and parameters of the `SigLipLoss` module include:
- `cache_labels`: Whether to cache labels for faster computation.
- `rank`: The rank of the current process when using multi-process training.
- `world_size`: The number of processes in multi-process training.
- `bidir`: Whether to use bidirectional communication during training.
- `use_horovod`: Whether to use Horovod for distributed training.

The SigLIP framework is based on the Sigmoid Loss for Language Image Pre-Training research paper, which provides more detailed information about the approach. You can find the paper [here](https://arxiv.org/abs/2303.15343).

---

## 3. Installation <a name="installation"></a>

Before using the `SigLipLoss` module, make sure you have the necessary dependencies installed. You can install the module using pip:

```bash
pip install sigliploss
```

---

## 4. Usage <a name="usage"></a>

In this section, we'll cover how to use the `SigLipLoss` module effectively.

### 4.1. Initializing SigLipLoss <a name="initializing-sigliploss"></a>

To use the `SigLipLoss` module, you first need to initialize it. You can provide optional parameters like `cache_labels`, `rank`, `world_size`, `bidir`, and `use_horovod` during initialization.

```python
from zeta.nn.modules import SigLipLoss

# Initialize SigLipLoss module
loss = SigLipLoss(
    cache_labels=False, rank=0, world_size=1, bidir=True, use_horovod=False
)
```

### 4.2. Calculating Loss <a name="calculating-loss"></a>

The primary purpose of the `SigLipLoss` module is to calculate the contrastive loss between image and text features. You'll need to provide image features, text features, `logit_scale`, and `logit_bias` to calculate the loss.

```python
# Example data
image_features = torch.randn(10, 128)
text_features = torch.randn(10, 128)
logit_scale = 1.0
logit_bias = None

# Calculate loss
outputs = loss(image_features, text_features, logit_scale, logit_bias)
print(outputs)
```

### 4.3. Multi-process Communication <a name="multi-process-communication"></a>

If you're using multi-process training, `SigLipLoss` provides options for communication between processes. The module can exchange text features between processes to facilitate training. Use the `rank`, `world_size`, `bidir`, and `use_horovod` parameters to configure this behavior.

---

## 5. Examples <a name="examples"></a>

Let's dive into some examples to demonstrate how to use the `SigLipLoss` module in practice.

### 5.1. Example 1: Initializing SigLipLoss <a name="example-1-initializing-sigliploss"></a>

In this example, we'll initialize the `SigLipLoss` module with default parameters.

```python
from zeta.nn.modules. import SigLipLoss

# Initialize SigLipLoss module
loss = SigLipLoss()
```

### 5.2. Example 2: Calculating Loss <a name="example-2-calculating-loss"></a>

Now, let's calculate the loss using sample image and text features.

```python
import torch
from zeta.nn.modules. import SigLipLoss

# Initialize SigLipLoss module
loss = SigLipLoss()

# Example data
image_features = torch.randn(10, 128)
text_features = torch.randn(10, 128)
logit_scale = 1.0
logit_bias = None

# Calculate loss
outputs = loss(image_features, text_features, logit_scale, logit_bias)
print(outputs)
```

### 5.3. Example 3: Multi-process Communication <a name="example-3-multi-process-communication"></a>

In a multi-process training scenario, you can configure `SigLipLoss` for communication between processes. Here's an example:

```python
from zeta.nn.modules. import SigLipLoss

# Initialize SigLipLoss module with multi-process settings
loss = SigLipLoss(rank=0, world_size=4, bidir=True, use_horovod=False)
```

---

## 6. Additional Information <a name="additional-information"></a>

- **SigLIP Framework**: SigLIP (Sigmoid Loss for Language Image Pre-Training) is a research framework for efficient language-image pre-training. Refer to the [research paper](https://arxiv.org/abs/2303.15343) for in-depth information.
- **Training**: The `SigLipLoss` module is designed for training models within the SigLIP framework.
- **Multi-process Training**: It provides options for communication between processes during multi-process training.

---

## 7. Conclusion <a name="conclusion"></a>

The `SigLipLoss` module is a critical component of the SigLIP framework, enabling efficient training of models for language-image pre-training tasks. This documentation provides a detailed guide on its usage, parameters, and examples to help you integrate it into your projects effectively.
