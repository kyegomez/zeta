# Class Name: Encoder

The `Encoder` class is a subclass of the AttentionLayers class used largely in transformer models for natural language processing tasks. It is intended to read and process inputs without an enforced causality - meaning it does not maintain an implied sequence or order in the data it processes. As such, the Encoder can utilize context from all directions and all inputs are independently centric in attention operations.

## Class Signature
```python
class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
```

## Now let us dive deeper into the Class functionalities and making use of it.

### Parameters

|Parameter| Type | Description |
|--|--|--|
|`kwargs`| *args | arbitrary keyword arguments passed for initialization | 


### Note
"Causal" should not be included in `kwargs`, as causality is not applicable for an Encoder.

`super().__init__(causal=False, **kwargs)` is used to pass all arguments to the parent class i.e., AttentionLayer, where `causal=False` - ensuring that the Encoder does not consider causality in the attention/subsequent operations.

# Example of Implementing your own custom Encoder:

Let's take an example of creating a basic encoder for a Transformer model -

```python
import torch.nn as nn
from zeta.structs import AttentionLayers

class MyEncoder(AttentionLayers):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__(d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = super().forward(x)
        return self.linear(x)
```
We built a custom encoder by extending the AttentionLayers, added a linear layer after the attention operations.

# Example Usage:

Firstly, let's initialize the model:
```python
model = MyEncoder(d_model=512, nhead=8, num_layers=6)
```
The model is initialized with the dimensions of model `d_model=512`, number of heads `nhead=8`, and the number of layers `num_layers=6`.

Now, let's define some dummy input data and pass it through the model:

```python
import torch

x = torch.randn(10, 32, 512)  # (sequence_length, batch_size, d_model)
output = model(x)  # forward pass
print(output.shape)  # torch.Size([10, 32, 512])
```
The method `forward()` computes the forward pass of our custom encoder model.

## Note

Remember, `Encoder` can be viewed as a wrapping layer around `AttentionLayers`, that ensures non-causal behaviour for the encoder in a Transformer. Hence, it is used typically for operations where the entire sequence is available for consideration - like in a Transformer's encoder, while predicting masked tokens based on surrounding context etc. 

As seen in the example, it is easy to extend the `Encoder` class and add additional layers or functionality, if required, depending upon specific use-cases.  

## Disclaimer:
 The class could change since the provided code is a snippet and might not represent the final form the `Encoder` class would take. This documentation is aimed at guiding understanding of the basic idea, intent, usage and extension of the `Encoder` class based on the short provided code snippet. For exact details, refer to the actual implementation in its entirety.


