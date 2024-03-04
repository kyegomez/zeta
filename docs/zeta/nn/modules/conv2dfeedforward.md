
# Conv2DFeedforward

The `Conv2DFeedforward` is a `torch.nn` module part of the `zeta.nn` library, designed to implement a Convolutional Feedforward network as proposed in Vision Attention Network (VAN) by Guo et al. The network operates on input data that represents a tensor fo shape (N, L, C), where N is the batch size, L is the sequence context length, and C is the input feature dimension.

Import Example:
```python
import torch

from zeta.nn import Conv2DFeedforward
```

The architecture of this module is designed to process multi-dimensional data with rows and columns, and it includes convolutional layers combined with multi-layer perceptron (MLP) architecture to process feature-containing input data in a feedforward fashion.

### Parameters:

| Args                    | Description                                                                                                                                     |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| dim                     | Integer parameter - Total number of input features of the given data.                                                                          |
| hidden_layer_multiplier | Integer parameter - The multiplier factor used to determine the number of hidden features defined as a multiple of the input feature dimension.  |
| dim_out                 | Optional Integer parameter - The total number of output features of the given data.                                                             |
| activation              | Object - The non-linear activation function. Default: GELU (Gaussian Error Linear Unit).                                                        |
| dropout                 | Float parameter - Determines the probability of dropout on the feedforward network's output. Default: 0.1                                       |
| \*args                  | Additional positional parameters.                                                                                                                |
| \*\*kwargs              | Additional keyword parameters.                                                                                                                   |

### Methods:

1. **init_weights(self, **kwargs)**
    Function to initialize weights of the module. The weights are initialized based on the original initialization proposed in the vision attention network paper and it allows to initialize from the outside as well.

    Example Usage:
    ```python
    conv = Conv2DFeedforward(256, 1, 256)
    conv.init_weights()
    ```

2. **forward(self, x: Tensor) -> Tensor**
    The forward function processes the input tensor through the convolutional feedforward neural network and returns the output tensor.

    Example Usage:
    ```python
    conv = Conv2DFeedforward(256, 1, 256)
    x = torch.randn(2, 64, 256)
    output = conv(x)
    print(output.shape)
    ```
    Expected Output:
    ```
    torch.Size([2, 64, 256])
    ```

The `Conv2DFeedforward` module uses a combination of convolutional layers and multi-layer perceptron to provide a sophisticated framework to process multi-dimensional data, particularly for image-related classification or localization problems.

For additional details and in-depth research on the underlying architectures and concepts associated with the Conv2DFeedforward module, refer to the official Vision Attention Network paper provided at _VAN_.
