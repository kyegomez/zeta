# BasicHebbianGRUModel Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Class Definition](#class-definition)
3. [Initialization](#initialization)
4. [Forward Pass](#forward-pass)
5. [Usage Examples](#usage-examples)
6. [Additional Information](#additional-information)

---

## 1. Introduction <a name="introduction"></a>

The `BasicHebbianGRUModel` is a PyTorch-based model designed for text-based tasks. It combines Hebbian learning with a GRU (Gated Recurrent Unit) layer to process sequential data. This model introduces non-linearity through the ReLU (Rectified Linear Unit) activation function.

### Purpose
- The model is designed to learn and represent patterns in sequential data, making it suitable for various natural language processing (NLP) tasks.
- It applies Hebbian learning to adaptively adjust weights based on input patterns, followed by GRU processing for sequential data handling.
- The ReLU activation function introduces non-linearity, enabling the model to capture complex relationships in the data.

### Key Features
- Hebbian learning for weight adaptation.
- GRU layer for sequential data processing.
- ReLU activation for non-linearity.

---

## 2. Class Definition <a name="class-definition"></a>

```python
class BasicHebbianGRUModel(nn.Module):
    """
    A basic Hebbian learning model combined with a GRU for text-based tasks.

    Parameters:
    - dim (int): Dimension of the input features.
    - hidden_dim (int): Dimension of the hidden state in the GRU.
    - output_dim (int): Dimension of the output features.
    """
```

The `BasicHebbianGRUModel` class has the following attributes and methods:

- `dim` (int): Dimension of the input features.
- `hidden_dim` (int): Dimension of the hidden state in the GRU.
- `output_dim` (int): Dimension of the output features.

---

## 3. Initialization <a name="initialization"></a>

To create an instance of the `BasicHebbianGRUModel`, you need to specify the dimensions of input, hidden state, and output features. Here's how you can initialize the model:

```python
dim = 512  # Dimension of the input features
hidden_dim = 256  # Dimension of the hidden state in the GRU
output_dim = 128  # Dimension of the output features
model = BasicHebbianGRUModel(dim, hidden_dim, output_dim)
```

---

## 4. Forward Pass <a name="forward-pass"></a>

The forward pass of the model processes input data through several stages:

1. It applies Hebbian update rules to the weights.
2. The data is then passed through a GRU layer.
3. A ReLU activation function is applied to introduce non-linearity.
4. Finally, the output is passed through a fully connected layer.

Here's how to perform a forward pass:

```python
# Assuming input_tensor is a 3D tensor of shape (B, Seqlen, dim)
output = model(input_tensor)
```

---

## 5. Usage Examples <a name="usage-examples"></a>

### Example 1: Model Initialization

```python
dim = 512
hidden_dim = 256
output_dim = 128
model = BasicHebbianGRUModel(dim, hidden_dim, output_dim)
```

### Example 2: Forward Pass

```python
# Assuming input_tensor is a 3D tensor of shape (B, Seqlen, dim)
output = model(input_tensor)
```

### Example 3: Accessing Model Parameters

```python
# Accessing model parameters (weights, GRU parameters, FC layer parameters)
model_weights = model.weights
gru_parameters = model.gru.parameters()
fc_parameters = model.fc.parameters()
```

---

## 6. Additional Information <a name="additional-information"></a>

### Tips for Effective Usage
- For optimal results, ensure that input data is properly preprocessed and normalized.
- Experiment with different hyperparameters, such as the dimensions of hidden states and output features, to fine-tune the model for your specific task.

### References
- [GRU Documentation](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- [ReLU Activation Function](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)

---

This documentation provides an overview of the `BasicHebbianGRUModel`, its purpose, usage, and key features. For more details on its implementation and advanced usage, refer to the source code and additional resources.
