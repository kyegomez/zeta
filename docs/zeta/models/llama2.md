# LLama2

## Class Overview

The class LLama2 is a custom transformer model built for Natural Language Processing (NLP) tasks. The objective of this class is to provide a compact yet powerful transformer model for the application of various NLP tasks, from translation to text generation and more.

The LLama2 transformer in this class provides a broad range of customizable parameters, allowing for it to be fine-tuned for specific tasks and datasets. It supports arguments for the sequence length, model dimensions, layer depths, number of heads, and several other options, providing extensive adaptability for various NLP tasks.

## Class Structure

```python
class LLama2:
    def __init__(
        self,
        num_tokens=50432,
        max_seq_len=8192,
        dim=2560,
        depth=32,
        dim_head=128,
        heads=24,
        rotary_xpos=True,
        attn_flash=True,
    ):
        super().__init__()

        self.llama2 = Transformer(
            num_tokens=50000,
            max_seq_len=4096,
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                dim_head=dim_head,
                heads=heads,
                attn_flash=attn_flash,
                rotary_xpos=rotary_xpos,
            ),
        )
        self.decoder = AutoregressiveWrapper(self.decoder)

    def forward(self, text):
        model_input = self.decoder.forward(text)[0]
        return self.decoder(model_input, padded_x=model_input[0])
```

Function Name: `__init__`

Purpose: Initializes the LLama2 class.

| Parameter | Data Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| num_tokens | int | 50432 | The total number of tokens in the input vocabulary. |
| max_seq_len | int | 8192 | The maximum sequence length that the model can accept. |
| dim | int | 2560 | The model's embedding dimensionality. |
| depth | int | 32 | The number of transformer layers in the model. |
| dim_head | int | 128 | The dimensionality of the head in the self-attention mechanism of the transformer model. |
| heads | int | 24 | The number of heads for the multi-head self attention mechanism of the transformer model. |
| rotary_xpos | bool | True | Whether to apply rotary positional embeddings to the input sequence. |
| attn_flash | bool | True | Whether to use the flash attention mechanism. |

Function Name: `forward`

Purpose: Defines the forward pass of the model.

| Parameter | Data Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| text | string | | The input text which the model processes. |

Returns: A tensor representation of model's output given the model_input.
    
## Usage Examples

### Example 1: Text Processing

This example illustrates how to instantiate the model and pass a sample text through it.

```python
import torch
from torch.nn import Decoder, Transformer

from zeta.models import LLama2
from zeta.structs import AutoregressiveWrapper

# Initializing model
llama2_model = LLama2()

# Cut-off long text or pad short text
text = torch.tensor([1, 2, 3, 4])

# Passing text through model
output = llama2_model.forward(text)

print(output)
```

### Example 2: Customizing Model Parameters

This example illustrates how to instantiate the model with custom parameters.

```python
llama2_model = LLama2(
    num_tokens=1000, max_seq_len=512, dim=512, depth=4, dim_head=64, heads=4
)

text = torch.tensor([1, 2, 3, 4])

output = llama2_model.forward(text)

print(output)
```

### Example 3: Sequence Classification

This example illustrates how you could use this model for a sequence classification task.

```python
llama2_model = LLama2(
    num_tokens=5000, max_seq_len=256, dim=128, depth=2, dim_head=32, heads=2
)

text_sequences = torch.tensor([[1, 2, 3, 4], [2, 3, 1, 4]])
target_sequences = torch.tensor([1, 0])  # 2 sequences, 1 for each sequence

outputs = llama2_model.forward(text_sequences)
loss = loss_function(outputs, target_sequences)
```
In this usage example, an instance of the LLama2 class is created using custom parameters. A tensor representing text sequences is passed to the model, and the output is computed. You would typically use a loss function suitable for classification tasks (like Cross-Entropy Loss) and compute the loss against some target sequences. 

Note: The provided code is a basic example and might require adjustments like adding an appropriate classifier layer at the end, depending on the specific task requirements.
