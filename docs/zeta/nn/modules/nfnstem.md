# NFNStem

The Zeta.nn.modules library is designed to accommodate the numerous layers and operations built in torch.nn layers, also this code provides support for different operations and custom layers, the code, and the accompanying documentation allow users to implement deep learning-based neural network architectures in Python. The purpose of the Zeta.nn.modules is to provide a collection of pre-written layers and operations that can be used to create new neural network architectures, making the process more efficient and less error-prone.

### Class Name: NFNStem

The `NFNStem` module represents the leaf node of the Neural Filter Network (NFN) architecture, aiding in the extraction of features and refining them through multiple layers of convolution.

#### Args:
| Argument       | Description                                     | Data Type | Default                              |
|----------------|-------------------------------------------------|-----------|--------------------------------------|
| in_channels    | Input channel sizes for each layer               | List[int] | [3, 16, 32, 64]                      |
| out_channels   | Output channel sizes for each layer              | List[int] | [16, 32, 64, 128]                    |
| kernel_size    | Size of the convolutional kernel                 | int       | 3                                    |
| stride         | Stride values for each convolutional layer       | List[int] | [2, 1, 1, 2]                         |
| activation     | Activation function after each convolution layer | nn.Module | nn.GELU()                            |

#### Usage Examples:
```python
import torch

from zeta.nn import NFNStem

# Create a random tensor with the shape of (1, 3, 224, 224)
x = torch.randn(1, 3, 224, 224)

# Instantiate the NFNStem module
model = NFNStem()

# Forward pass
out = model(x)
print(out.shape)
# Output: torch.Size([1, 128, 28, 28])
```
```python
# Creating a custom NFNStem
nfn_stem = NFNStem(
    in_channels=[5, 10, 15, 20], out_channels=[10, 20, 30, 40], activation=nn.ReLU()
)
feature_map = nfn_stem(input_data)
print(feature_map.shape)
```
```python
import torch

from zeta.nn import NFNStem

# Utilization of NFNStem with custom parameters
stem = NFNStem(in_channels=[4, 8, 16, 16], out_channels=[8, 16, 32, 64])
data = torch.randn(1, 4, 128, 128)
output = stem(data)
print(output.shape)
```

The main purpose of the `NFNStem` class is to allow the construction of a sequence of neural network layers to process input data. The `forward` method takes an input tensor `x` and processes it through several convolution and activation layers, returning the output tensor.

Additional information and tips:
- Ensure that the input tensor has the appropriate shape and data type compatible with the individual layers.
- The parameters such as `in_channels`, `out_channels`, `kernel_size`, and `stride` can be fine-tuned based on the specific requirements of the neural network architecture.

Include references and resources:
- Further insights into the "Neural Filter Network" architecture can be explored at [Link to research paper].
- The official repository for Zeta.nn.modules can be found at [Link to Zeta.nn.modules repository].

By following this documented approach, the users can efficiently understand, implement and customize the Zeta.nn.modules for their specific neural network architecture needs.
