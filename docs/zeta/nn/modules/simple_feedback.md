# SimpleFeedForward: Feedforward Neural Network with LayerNorm and GELU Activations

**Overview and Introduction**

The `SimpleFeedForward` function is a utility function that creates a feedforward neural network architecture with layer normalization (`LayerNorm`) and Gaussian Error Linear Unit (`GELU`) activations. The architecture is particularly well-suited for applications in deep learning where input feature normalization and non-linear transformations are essential for effective model training and generalization.

**Main Features**:

- **Layer Normalization**: Normalizes the input data across the feature dimension, ensuring that the input to each subsequent layer has a stable distribution. This aids in faster and more stable convergence during training.
  
- **GELU Activation**: A smooth activation function that is used for better performance in deeper architectures, especially transformer models.
  
- **Dropout**: A regularizing technique where randomly selected neurons are ignored during training, reducing overfitting and improving model generalization.

---

**Function Definition**:

```python
def SimpleFeedForward(dim: int, hidden_dim: int, dropout: float = 0.1) -> nn.Sequential:
```

**Parameters**:

| Parameter    | Type  | Default | Description                                  |
|--------------|-------|---------|----------------------------------------------|
| `dim`        | int   | --      | Input dimension of the neural network.       |
| `hidden_dim` | int   | --      | Hidden layer dimension of the neural network.|
| `dropout`    | float | 0.1     | Dropout probability for regularization.      |

---

**Functionality and Usage**:

The `SimpleFeedForward` function constructs a neural network that consists of the following sequence of operations:
1. Layer normalization of the input features.
2. A linear transformation that expands the input to a specified hidden dimension.
3. GELU activation function.
4. Another linear transformation that maps the hidden layer back to the original input dimension.
5. Dropout for regularization.

This particular sequence ensures that the neural network can learn a rich representation from the input features while being regularized to prevent overfitting.

**Usage Examples**:

1. **Basic Usage**:

   ```python
   import torch
   import torch.nn as nn

   from zeta.nn.modules import SimpleFeedForward

   model = SimpleFeedForward(768, 2048, 0.1)
   x = torch.randn(1, 768)
   output = model(x)
   print(output.shape)  # torch.Size([1, 768])
   ```

2. **Integrating with Other Architectures**:

   ```python
   import torch
   import torch.nn as nn

   from zeta.nn.modules import SimpleFeedForward


   class CustomModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.ff = SimpleFeedForward(768, 2048, 0.1)
           self.final_layer = nn.Linear(768, 10)  # Example output layer

       def forward(self, x):
           x = self.ff(x)
           return self.final_layer(x)


   model = CustomModel()
   x = torch.randn(1, 768)
   output = model(x)
   print(output.shape)  # torch.Size([1, 10])
   ```

3. **Using Different Dropout Values**:

   ```python
   import torch
   import torch.nn as nn

   from zeta.nn.modules import SimpleFeedForward

   model = SimpleFeedForward(768, 2048, 0.5)  # Setting a higher dropout value
   x = torch.randn(1, 768)
   output = model(x)
   print(output.shape)  # torch.Size([1, 768])
   ```

---

**Additional Information and Tips**:

- For tasks where overfitting is a concern, consider increasing the `dropout` parameter value to introduce more regularization.
  
- The function returns an `nn.Sequential` model, making it easy to integrate into larger architectures or pipelines.
  
- Remember that the effective capacity of the model is determined by the `hidden_dim` parameter. Adjusting this can help in balancing model complexity and performance.

---

**References and Resources**:

- [Layer Normalization Paper](https://arxiv.org/abs/1607.06450)
  
- [GELU Activation Function](https://arxiv.org/abs/1606.08415)

---

