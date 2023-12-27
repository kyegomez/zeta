# GPT4 Class

GPT4 is a class providing the architecture of a transformer-based model. The class primarily consists of two main components, a Transformer and an AutoregressiveWrapper. 

Based on the method used by OpenAI's GPT-3, the GPT4 in this implementation expands on that base with user-specified or default parameters. These parameters allow users to customize the architecture, depth, and functionality of their models for specific use-cases.

## Initialize the class

The class is initialized by the following arguments:

| Argument                     | Type     | Default | Description |
| -----------------------------| -------- | ------- | ----------- |
| num_tokens                   | int      | 50432   | Number of tokens in the vocabulary |
| max_seq_len                  | int      | 8192    | Maximum length of the sequence |
| dim                          | int      | 2560    | Dimension of the model |
| depth                        | int      | 32      | Depth of the model |
| dim_head                     | int      | 128     | Dimension of the model head |
| heads                        | int      | 24      | Number of heads |
| use_abs_pos_emb              | bool     | False   | Whether to use absolute position embedding |
| alibi_pos_bias               | bool     | True    | Alibi position bias |
| alibi_num_heads              | int      | 12      | Number of alibi heads |
| rotary_xpos                  | bool     | True    | Rotary position |
| attn_flash                   | bool     | True    | Attention flash |
| attn_one_kv_head             | bool     | True    | Attention one key/value head for multiquery attention |
| qk_norm                      | bool     | True    | Query-key normalization |
| attn_qk_norm                 | bool     | True    | Attention query-key normalization |
| attn_qk_norm_dim_scale       | bool     | True    | Attention query-key normalization dimension scale |

Each of these arguments can be modified to suit specific needs of the user. 

## Implementing the transformer class

The Transformer architecture used in the GPT4 model forms the backbone of the class. It utilizes an attention mechanism to focus on different words in a sequence while processing the input data.

In this case, the Transformer is a Decoder, which transpires the depth, dim_head, heads, alibi_pos_bias, alibi_num_heads, rotary_xpos, attn_flash, attn_one_kv_head, qk_norm, attn_qk_norm, and attn_qk_norm_dim_scale properties from the GPT4 arguments.

If initialization fails for any reason, an exception is caught and logged in the console, and the exception is re-raised.

## AutoregressiveWrapper

As a next step, the transformer is wrapped with an AutoregressiveWrapper. Autoregressive models are ones where the output from one step is fed as an input to the next step. This allows for modeling the sequence of data effectively, thus making it excellent for tasks like text generation and language modelling.

## Forward function

The `forward` function of the GPT4 class starts by taking `text_tokens` as input. This variable represents the tokenized input sentences.

In the forward function, a Transformer (loaded by the decoder) is applied to forward `text_tokens`. The result is a `model_input` variable, which is then passed into the decoder along with the `padded_x` parameter.

If exceptions occur during the forward pass, they are caught and logged in the console, and the exception is re-raised.

## Usage

Here's how you can use the GPT4 class:

```python
import torch
from torch import nn
from zeta.models import GPT4

# Initialize with default parameters
model = GPT4()

# Representing 3 sequences of the maximum length of 8192
input = torch.randint(0, 50432, (3, 8192))

# Pass the input to the model's forward method
output = model.forward(input)
```

## Conclusion

The GPT4 class is a powerful tool for creating Transformer-based language models. With the flexibility it provides, users can customize the model per their requirements and specifications. Whether it be altering the dimensionality, the number of heads in multihead attention, or whether to use absolute position embeddings, the GPT4 class provides a versatile and flexible architecture for your next natural language processing project.
