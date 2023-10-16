# `StableAdamWUnfused` Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Class: `StableAdamWUnfused`](#class-stableadamwunfused)
   - [Initialization](#initialization)
   - [Key Functions](#key-functions)
3. [Usage Examples](#usage-examples)
   - [Training a Deep Learning Model](#training-a-deep-learning-model)
   - [Using Custom Floating Point Precision](#using-custom-floating-point-precision)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
4. [Additional Information](#additional-information)
   - [StableAdamW Algorithm](#stableadamw-algorithm)
   - [Setting Precision to "custom_fp16"](#setting-precision-to-custom_fp16)
5. [References](#references)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the documentation for the `StableAdamWUnfused` optimizer in the Shapeless library! `StableAdamWUnfused` is designed to provide a stable and efficient implementation of the AdamW optimizer with optional features like custom floating point precision and update clipping. 

### Key Features
- StableAdamW: Stable implementation of the AdamW optimizer.
- Custom Floating Point Precision: Choose between standard precision and custom precision for gradients.
- Update Clipping: Apply update clipping to prevent excessively large updates.

In this documentation, you will learn how to use the `StableAdamWUnfused` optimizer effectively, understand its architecture, and explore examples of its applications.

---

## 2. Class: `StableAdamWUnfused` <a name="class-stableadamwunfused"></a>

The `StableAdamWUnfused` class is the core component of the Shapeless library, providing advanced optimization techniques for deep learning models. Below, we'll delve into its initialization and key functions.

### Initialization <a name="initialization"></a>

```python
optimizer = StableAdamWUnfused(
    params,
    lr=0.002,
    weight_decay=0.2,
    betas=(0.9, 0.99),
    eps=1e-8,
    clip_thresh=1.0,
    precision="amp_bfloat16",
    custom_scalar=65536,
)
```

#### Parameters:
- `params` (iterable): Model parameters for optimization.
- `lr` (float): Learning rate (default: 0.002).
- `weight_decay` (float): Weight decay (L2 penalty) (default: 0.2).
- `betas` (Tuple[float, float]): Coefficients for computing running averages of gradient and its square (default: (0.9, 0.99)).
- `eps` (float): Small constant to prevent division by zero (default: 1e-8).
- `clip_thresh` (float): Threshold for update clipping (default: 1.0).
- `precision` (str): Precision mode ("amp_bfloat16" or "custom_fp16") (default: "amp_bfloat16").
- `custom_scalar` (int): Custom scalar for gradients (default: 65536).

### Key Functions <a name="key-functions"></a>

#### `step(closure=None)`
Performs a single optimization step. Computes gradients and updates model parameters.

- `closure` (Optional[Callable]): A closure that computes the loss (default: None).

---

## 3. Usage Examples <a name="usage-examples"></a>

Now, let's explore practical examples of using the `StableAdamWUnfused` optimizer in various scenarios.

### Training a Deep Learning Model <a name="training-a-deep-learning-model"></a>

```python
# Initialize the optimizer
optimizer = StableAdamWUnfused(model.parameters(), lr=0.001, weight_decay=0.0001)

# Inside the training loop
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

### Using Custom Floating Point Precision <a name="using-custom-floating-point-precision"></a>

```python
# Initialize the optimizer with custom_fp16 precision
optimizer = StableAdamWUnfused(model.parameters(), lr=0.001, precision="custom_fp16")

# Inside the training loop, use custom scalar
custom_scalar = 65536  # Custom scalar value
(loss * custom_scalar).backward()  # Backward pass with custom scalar
optimizer.step()
```

### Hyperparameter Tuning <a name="hyperparameter-tuning"></a>

```python
# Define a grid of hyperparameters
learning_rates = [0.001, 0.01, 0.1]
weight_decays = [0.0001, 0.001, 0.01]

# Loop through hyperparameters
for lr in learning_rates:
    for wd in weight_decays:
        optimizer = StableAdamWUnfused(model.parameters(), lr=lr, weight_decay=wd)
        
        # Training and evaluation code here
```

These examples showcase how the `StableAdamWUnfused` optimizer can be used in training deep learning models with various configurations.

---

## 4. Additional Information <a name="additional-information"></a>

### StableAdamW Algorithm <a name="stableadamw-algorithm"></a>

The `StableAdamWUnfused` optimizer implements the StableAdamW algorithm, which is a stable version of the AdamW optimizer. It provides stability and efficiency in deep learning optimization.

### Setting Precision to "custom_fp16" <a name="setting-precision-to-custom_fp16"></a>

You can set the precision mode to "custom_fp16" to use a custom scalar value for gradients. This mode allows fine-grained control over the precision of gradient calculations.

---

## 5. References <a name="references"></a>

For further information and research papers related to the `StableAdamWUnfused` optimizer and its stability improvements, please refer to the following resources:

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Custom Floating Point Precision](https://example.com/custom-precision-paper)

Explore these references to gain a deeper understanding of the optimization techniques implemented in `StableAdamWUnfused`.

Feel free to reach out to the Shapeless community for any questions or discussions regarding this optimizer. Happy optimizing!
