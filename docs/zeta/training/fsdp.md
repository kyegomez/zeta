# `fsdp` Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Function: `fsdp`](#function-fsdp)
   - [Initialization](#initialization)
   - [Parameters](#parameters)
   - [Mixed Precision Modes](#mixed-precision-modes)
   - [Sharding Strategies](#sharding-strategies)
3. [Usage Examples](#usage-examples)
   - [Basic FSDP Wrapper](#basic-fsdp-wrapper)
   - [Automatic Layer Wrapping](#automatic-layer-wrapping)
   - [Advanced Configuration](#advanced-configuration)
4. [Additional Information](#additional-information)
   - [FullyShardedDataParallel (FSDP)](#fullyshardeddataparallel-fsdp)
   - [Mixed Precision Training](#mixed-precision-training)
   - [Model Sharding](#model-sharding)
5. [References](#references)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the documentation for the Zeta library! Zeta provides a powerful utility function, `fsdp`, that wraps a given PyTorch model with the FullyShardedDataParallel (FSDP) wrapper. This enables efficient data parallelism and model sharding for deep learning applications.

### Key Features
- **Efficient Data Parallelism**: FSDP allows you to efficiently parallelize training across multiple GPUs.
- **Mixed Precision Training**: Choose between BFloat16 (bf16), Float16 (fp16), or Float32 (fp32) precision modes.
- **Model Sharding**: Apply gradient sharding, full model sharding, or no sharding based on your needs.

In this documentation, you will learn how to use the `fsdp` function effectively, understand its architecture, and explore examples of its applications.

---

## 2. Function: `fsdp` <a name="function-fsdp"></a>

The `fsdp` function is the core component of the Zeta library, providing a straightforward way to wrap your PyTorch model with FSDP for efficient distributed training.

### Initialization <a name="initialization"></a>

```python
model = fsdp(
    model, auto_wrap=False, mp="fp32", shard_strat="NO_SHARD", TransformerBlock=None
)
```

### Parameters <a name="parameters"></a>

- `model` (torch.nn.Module): The original PyTorch model to be wrapped with FSDP.
- `auto_wrap` (bool, optional): If True, enables automatic wrapping of the model's layers based on the `transformer_auto_wrap_policy`. Default is False.
- `mp` (str, optional): The mixed precision mode to be used. Can be 'bf16' for BFloat16, 'fp16' for Float16, or 'fp32' for Float32 precision. Default is 'fp32'.
- `shard_strat` (str, optional): The sharding strategy to be used. Can be 'SHARD_GRAD' for sharding at gradient computation, 'FULL_SHARD' for full model sharding, or 'NO_SHARD' for no sharding. Default is 'NO_SHARD'.
- `TransformerBlock` (Type, optional): A custom transformer layer type. Only used if `auto_wrap` is True.

### Mixed Precision Modes <a name="mixed-precision-modes"></a>

- `bf16` (BFloat16): Lower precision for faster training with minimal loss in accuracy.
- `fp16` (Float16): Higher precision than BFloat16 but still faster than full precision.
- `fp32` (Float32): Full single-precision floating-point precision.

### Sharding Strategies <a name="sharding-strategies"></a>

- `SHARD_GRAD` (Sharding at Gradient Computation): Shards gradients during the backward pass.
- `FULL_SHARD` (Full Model Sharding): Shards the entire model for parallelism.
- `NO_SHARD` (No Sharding): No sharding, suitable for single-GPU training.

---

## 3. Usage Examples <a name="usage-examples"></a>

Now, let's explore practical examples of using the `fsdp` function in various scenarios.

### Basic FSDP Wrapper <a name="basic-fsdp-wrapper"></a>

```python
import torch.nn as nn

# Define your PyTorch model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# Wrap the model with FSDP using default settings (no sharding, fp32 precision)
fsdp_model = fsdp(model)
```

### Automatic Layer Wrapping <a name="automatic-layer-wrapping"></a>

```python
import torch.nn as nn


# Define a custom transformer layer type
class TransformerBlock(nn.Module):
    def __init__(self):
        # Define your custom transformer layer here
        pass


# Define your PyTorch model with transformer layers
model = nn.Sequential(
    nn.Linear(784, 256),
    TransformerBlock(),
    nn.Linear(256, 10),
)

# Wrap the model with FSDP and enable automatic layer wrapping
fsdp_model = fsdp(model, auto_wrap=True, TransformerBlock=TransformerBlock)
```

### Advanced Configuration <a name="advanced-configuration"></a>

```python
import torch.nn as nn

# Define your PyTorch model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# Wrap the model with FSDP with custom settings (full model sharding, bf16 precision)
fsdp_model = fsdp(model, mp="bf16", shard_strat="FULL_SHARD")
```

These examples demonstrate how to use the `fsdp` function to wrap your PyTorch models with FSDP for distributed training with various configurations.

---

## 4. Additional Information <a name="additional-information"></a>

### FullyShardedDataParallel (FSDP) <a name="fullyshardeddataparallel-fsdp"></a>

FSDP is a powerful wrapper that enables efficient data parallelism and model sharding. It optimizes gradient communication and memory usage during distributed training.

### Mixed Precision Training <a name="mixed-precision-training"></a>

Mixed precision training involves using lower-precision data types for certain parts of the training pipeline, leading to faster training times with minimal loss in accuracy.

### Model Sharding <a name="model-sharding"></a>

Model sharding is a technique used to distribute model parameters across multiple devices or GPUs, improving training speed and memory efficiency.

---

## 5. References <a name="references"></a>

For further information and research papers related to FSDP, mixed precision training, and model sharding, please refer to the following resources:

- [FSDP Documentation](https://example.com/fsdp-docs)
- [Mixed Precision Training in Deep Learning](https://example.com/mixed-precision-paper)
- [Efficient Model Parallelism for Deep Learning](https://example.com/model-sharding-paper)

Explore these references to gain a deeper understanding of the techniques and concepts implemented in the Zeta library and the `fsdp` function.

Feel free to reach out to

 the Zeta community for any questions or discussions regarding this library. Happy deep learning!