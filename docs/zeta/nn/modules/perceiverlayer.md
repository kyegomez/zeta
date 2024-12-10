# Perceiver Layer

Multi-head attention mechanism often works well in analyzing subspaces of information, and the PerceiverLayer class is a constituent layer of a general-purpose architecture called the Perceiver, which uses multi-head attention mechanisms to analyze subspaces of information. It consists of a self-attention module followed by cross-attention and a feed-forward network.

The PerceiverLayer class takes in three inputs: query, key, and value tensors, and applies a series of operations using attention and a feed-forward layer to yield an output tensor with the same input tensor dimensions. Some of the key parameters for the class include the dimension of the input tensor, number of heads for multi-head attention, number of layers, dimensions of each attention head, dropout rates, and other parameters that define the architecture.

```python
Args[]
| arg  | description | type | default
|-------|-------------|------|---------
| dim | dimension of the input tensor | int | -
| heads | number of heads | int | -
| depth | number of layers | int | -
| dim_head | dimension of each head | int | 64
| dropout | dropout rate | float | 0.1
| ff_dropout | feed forward dropout rate | float | 0.1
| ff_mult | feed forward multiplier | int | 4

Examples

Creating an instance of the PerceiverLayer class and applying it to query, key, and value tensors:
```python
import torch
from zeta.nn import PerceiverLayer

q = torch.randn(1, 32, 512)
k = torch.randn(1, 32, 512)
v = torch.randn(1, 32, 512)
layer = PerceiverLayer(512, 8, 6, 64)
print(layer(q, k, v).shape)
```
Expected Output:
``` python
torch.Size([1, 32, 512])
```

The above example demonstrates the basic usage of the PerceiverLayer class by creating an instance and applying it to input tensors.

The multi-head attention operation within the PerceiverLayer class operates by taking the query tensor and then sending the output into the query of the cross-attention, where the cross-attention takes in the key and value tensors. The output of the cross-attention is then sent into a feed-forward layer to generate the output tensor.

The self_attn layer is used to perform self-attention on the query tensor, followed by concatenation of key and value tensors, and then input to the cross-attn layer for cross-attention, and finally, the feed-forward layer is applied. This process helps the model to process and understand the information across different dimensions.

The forward method of the PerceiverLayer applies the attention and feed-forward layer to input tensors:
```python
def forward(
    self,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
):
```

In this method, the query, key, and value tensors are passed as input, and a mask tensor can also be provided. The shapes of input tensors are specified in the parameter descriptions to ensure the correct input to this method. The comment above the method explains the high-level description of what this method does, including the input arguments and their shapes.

The PerceiverLayer class provides the capability to understand and process large scale and high-dimensional data using multi-head attention and a feed-forward architecture, which is particularly useful for tasks like image and video understanding, as well as language processing.

Utilizing this class to create custom attention-based models for applications such as visual recognition, natural language understanding, and generative modeling, can significantly benefit from the subtle interplay of attention mechanisms and feed-forward structures enabled by the PerceiverLayer class. Therefore, understanding the parameters, methods, and usage examples of this class are key to tapping its benefits effectively.

Finally, the PerceiverLayer class provides a great level of flexibility and adaptability to build complex models without worrying about attention mechanism implementation details.

Overall, the PerceiverLayer class is a vital component in building sophisticated and advanced models, which are capable of effectively processing and understanding high-dimensional and complex data across different domains. The class efficiently handles the design and managing of multi-head attention and a feed-forward layer architecture, which can be extensively used in various applications. Hence, the documentation and understanding of this class become essential to utilize its full potential.


In conclusion, the documentation for the PerceiverLayer is presented in this template, following the best practices of documentation for the PerceiverLayer class, including the thorough description of class, parameters, and methods. Additionally, it provides a clear and detailed explanation of class usage, accompanied by the usage examples to illustrate its usage and the expected outputs. After understanding the given documentation, one can create, understand, and leverage the features of this class to build complex models and solve real-world problems effectively.




