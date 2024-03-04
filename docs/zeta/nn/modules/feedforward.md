# `FeedForward`

## Overview

The `FeedForward` module is a feedforward neural network with LayerNorms and activation functions, designed for various transformer-based models. It offers flexibility in terms of the activation functions used, allowing you to choose between GELU, SiLU, or ReLU squared. Additionally, it supports the Gated Linear Unit (GLU) activation and LayerNorm (LN) after the activation layer for advanced configurations.

## Class Definition

```python
class FeedForward(nn.Module):
    """
    Feedforward neural network with LayerNorms and GELU activations

    Args:
        dim (int): Input dimension.
        dim_out (int, optional): Output dimension. Defaults to None (same as input dimension).
        mult (int, optional): Multiplier for the hidden dimension. Defaults to 4.
        glu (bool, optional): Whether to use the Gated Linear Unit (GLU) activation. Defaults to False.
        glu_mult_bias (bool, optional): Whether to use a bias term with the GLU activation. Defaults to False.
        swish (bool, optional): Whether to use the SiLU activation. Defaults to False.
        relu_squared (bool, optional): Whether to use the ReLU squared activation. Defaults to False.
        post_act_ln (bool, optional): Whether to apply LayerNorm after activation. Defaults to False.
        dropout (float, optional): Dropout probability. Defaults to 0.1 .
        no_bias (bool, optional): Whether to use bias terms in linear layers. Defaults to False.
        zero_init_output (bool, optional): Whether to initialize the output linear layer to zero. Defaults to False.

    Usage:
    >>> model = FeedForward(768, 2048, 0.1)
    >>> x = torch.randn(1, 768)
    >>> model(x).shape
    """
```

## Parameters

| Parameter Name   | Description                                               | Default Value  | Type   |
| -----------------|-----------------------------------------------------------|-----------------|--------|
| dim              | Input dimension                                           | -               | int    |
| dim_out          | Output dimension (optional)                               | None            | int    |
| mult             | Multiplier for hidden dimension                           | 4               | int    |
| glu              | Whether to use GLU activation                             | False           | bool   |
| glu_mult_bias    | Whether to use bias term with GLU activation              | False           | bool   |
| swish            | Whether to use SiLU activation                            | False           | bool   |
| relu_squared     | Whether to use ReLU squared activation                     | False           | bool   |
| post_act_ln      | Whether to apply LayerNorm after activation               | False           | bool   |
| dropout          | Dropout probability                                       | 0.1             | float  |
| no_bias          | Whether to use bias terms in linear layers                | False           | bool   |
| zero_init_output | Whether to initialize the output linear layer to zero     | False           | bool   |

## Usage Examples

### Example 1: Basic FeedForward Layer

```python
model = FeedForward(768, 2048, 0.1)
x = torch.randn(1, 768)
output = model(x)
print(output.shape)
```

### Example 2: Using SiLU Activation

```python
model = FeedForward(512, 1024, swish=True)
x = torch.randn(1, 512)
output = model(x)
print(output.shape)
```

### Example 3: Advanced Configuration with GLU Activation and LayerNorm

```python
model = FeedForward(256, 512, glu=True, post_act_ln=True, dropout=0.2)
x = torch.randn(1, 256)
output = model(x)
print(output.shape)
```

## Functionality

The `FeedForward` module performs a feedforward operation on the input tensor `x`. It consists of a multi-layer perceptron (MLP) with an optional activation function and LayerNorm. The exact configuration depends on the parameters provided during initialization.

The key steps of the forward pass include:
1. Projection of the input tensor `x` to an inner dimension.
2. Application of the specified activation function (e.g., GELU, SiLU, or ReLU squared).
3. Optionally, LayerNorm is applied after the activation.
4. Dropout is applied for regularization.
5. Finally, a linear transformation maps the inner dimension to the output dimension.

The `FeedForward` module offers flexibility in choosing activation functions, enabling you to experiment with different configurations in transformer-based models.

## Tips and Considerations

- Experiment with different activation functions to find the best configuration for your model.
- Adjust the dropout rate to control overfitting.
- Consider using LayerNorm for improved performance, especially in deep networks.
- The `zero_init_output` option can be useful for certain initialization strategies.
