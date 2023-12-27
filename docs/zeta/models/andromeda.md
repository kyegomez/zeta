# Class Name: Andromeda
**Module Description**

This documentation provides details on the functionality of the Andromeda class from the zeta.models library. 

The Andromeda class is a transformer-based model helper class that acts as a wrapper for the Transformer and AutoregressiveWrapper modules, defaulting or accepting user-specified values in its configuration. 

Features of the Andromeda model include but are not limited to: 
- Configurable model dimensions, including token count, maximum sequence length, layer depth, and head dimensions.
- Abstract position embeddings, alibi position biases, rotary positions, attentions, and buffer elements which are all modifiable by the user.

## Class Definition:

```python
class Andromeda(Module):
    """
    Andromeda is a transformer-based model architecture. It initializes with
    a Transformer and AutoregressiveWrapper with default or user-specified parameters.
    """
```
This class inherits the PyTorch Module class and serves as a wrapper to both the Transformer and AutoregressiveWrapper classes. 

## Initialization (__init__) Function:
The init function is where the Transformer and AutoregressiveWrapper objects are assigned to `self.Andromeda` and `self.decoder` respectively. 

```python
 def __init__(
        self,
        num_tokens=50432,
        max_seq_len=8192,
        dim=2560,
        depth=32,
        dim_head=128,
        heads=24,
        use_abs_pos_emb=False,
        alibi_pos_bias=True,
        alibi_num_heads=12,
        rotary_xpos=True,
        attn_flash=True,
        attn_kv_heads=2,
        qk_norm=True,
        attn_qk_norm=True,
        attn_qk_norm_dim_scale=True,
    ):
```

The parameters and their defaults used in initialization are listed below

| Parameter | Default Value | Description |
| ------------- | ------------- | ------------- |
| num_tokens | 50432 | Number of tokens in the vocabulary |
| max_seq_len  | 8192 | Maximum sequence length |
| dim  | 2560 | Dimension of the model |
| depth  | 32 | Depth of the model |
| dim_head  | 128 | Dimension of the model head |
| heads | 24 | Number of heads |
| use_abs_pos_emb  | False | Whether to use absolute position embedding |
| alibi_pos_bias  | True | Alibi position bias |
| alibi_num_heads  | 12 | Number of alibi heads |
| rotary_xpos | True | Rotary position |
| attn_flash | True | Attention flash |
| attn_kv_heads | 2 | Number of attention key/value heads |
| qk_norm | True | Query-key normalization |
| attn_qk_norm | True | Attention query-key normalization |
| attn_qk_norm_dim_scale | True | Attention query-key normalization dimension scale |

## Forward Function
Forward propagation in PyTorch involves defining the computation performed at every call. In the Andromeda class, this computation involves passing input text tokens through the decoder. If an exception occurs during this forward propagation, an error message will be printed and an exception will be thrown.

```python
 def forward(self, text_tokens, **kwargs):
        """
        Forward pass through the model. It expects the input text_tokens.
        """
 ```
The parameters used in forward function are listed below:

| Parameter | Description |
| ------------- | ------------- |
| text_tokens | Input tokens |
| **kwargs | Other arguments |

The forward function returns the output from the decoder.

## Code Example:
Below is a simple example of instantiating the Andromeda class and using it for forward propagation:

```python
# Import necessary libraries and modules
from torch.nn import Module
from zeta.models import Andromeda

# Initialize the Andromeda class with default parameters
model = Andromeda()

# Define your input text tokens
text_tokens = torch.randn(1, 8192)

# Perform forward pass through the model
output = model.forward(text_tokens)
```

**Note** 
Techniques such as query-key normalization aid in the alignment of the query’s distribution to that of the key, in order to reduce the negative impacts of any input with a wildly different distribution. As such, the parameters related to normalization (qk_norm, attn_qk_norm, attn_qk_norm_dim_scale) default to True, but can be toggled off based on the specific needs of your application.

Also, It's important to ensure that the defined text tokens fit within the dimensions defined for `num_tokens` and `max_seq_len`. Otherwise, you might encounter an error during forward pass. 

For more information on the underlying Transformer and AutoregressiveWrapper modules, please check the official PyTorch documentation. 

## Other Additional Information & Tips 
The Andromeda class is notable for its robust set of flexible features that can lend it to varying use-cases and it is inherently versatile due to its Transformer and AutoregressiveWrapper architecture. This model emphasizes on the detail to accepting user-specified parameters for a high level of customization. 

However, due to its complexity and high-dimensional nature, this model may not be preferable under constraints of memory, processing power or the need for simplicity. 

## References & External Resources

- [Official PyTorch Docs](https://pytorch.org/docs/stable/nn.html) for more information on underlying classes and modules.
- [Understanding Transformers in NLP](https://towardsdatascience.com/transformers-141e32e69591) for conceptual knowledge on Transformer models.
- [Autoregressive Models](https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/) for understanding on autoregressive models. 

Enjoy exploring the Andromeda class from the zeta.models library!
