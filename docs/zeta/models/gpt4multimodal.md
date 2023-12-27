# GPT4MultiModal

The `GPT4MultiModal` class is a subclass of the `torch.nn.Module` class. This class serves as a model for handling both image and text input in the form of sequences. It integrates the ViTransformerWrapper for image encoding and the Transformer for text decoding.

The primary aim of this class is to enable encoding an image and use it as context for generating a text sequence, hence the name `GPT4MultiModal`. Typical usage would be to pass an image to the encoder and a sequence of tokens (corresponding to a language prompt) to the decoder. The class will output a sequence of tokens- the length of the sequence will depend on the transformer architecture used.

## Class Constructor
This class accepts the following parameters:

| Parameters | Keyboard Argument | Type | Default Value | Description |
|:-------------:|:------:|:--------:|:---------------:|:------------:|
| image_size| image_size | int | 256 | Input image size |
| patch_size | patch_size | int | 32 | Size of each image patch |
| encoder_dim | encoder_dim | int | 512 | Dimension of encoder |
| encoder_depth | encoder_depth | int | 6 | The depth of the encoder |
| encoder_heads | encoder_heads | int | 8 | The number of attention heads in the encoder |
| num_tokens | num_tokens | int | 20000 | The number of unique tokens |
| max_seq_len | max_seq_len | int | 1024 | Maximum sequence length for text |
| decoder_dim | decoder_dim | int | 512 | Dimension of decoder |
| decoder_depth | decoder_depth | int | 6 | The depth of the decoder |
| decoder_heads | decoder_heads | int | 8 | The number of attention heads in the decoder |
| alibi_num_heads | alibi_num_heads | int | 4 | The number of attention heads per transformer |
| use_abs_pos_emb| use_abs_pos_emb | bool | False | If True, embeds input using absolute positional embedding |
| cross_attend | cross_attend | bool | True | If True, enables cross attention in decoder |
| alibi_pos_bias | alibi_pos_bias | bool | True | If True, positional bias is added to alibi |
| rotary_xpos | rotary_xpos | bool | True |Enables rotary positional embeddings |
| attn_flash | attn_flash | bool | True | If True, enables the use of Flash-like attention |
| qk_norm | qk_norm | bool | True | If True, enables query-key normalization |

## Methods
The following methods are available in this class.

#### `forward(self, img, text) -> Union[Tensor, str]`
The `forward` method is used to perform the forward propagation operation of the GPT4MultiModal model. It accepts an image and a sequence of tokens and returns a sequence of tokens.

Parameters:

| Parameters | Keyboard Argument | Type | Default Value | Description |
|:-------------:|:------:|:--------:|:---------------:|:------------:|
| img | img | Tensor | - | The input image tensor |
| text | text | Tensor | - | The sequence of tokens to be used as input |

Returns:

| Type | Description |
|:--------:|:------------:|
| Union[Tensor, str] | Output sequence of tokens or an error message if an exception is encountered |

# Example of Use

Consider having an image tensor `img` of size (1, 256, 256, 3) and a text tensor `text` of size (1, 50). Here is an example of how to use `GPT4MultiModal`

```python
import torch
from zeta.models import GPT4MultiModal

# Initialize the model
model = GPT4MultiModal(image_size=256, 
                       patch_size=32, 
                       encoder_dim=512, 
                       encoder_depth=6, 
                       encoder_heads=8, 
                       num_tokens=20000, 
                       max_seq_len=1024, 
                       decoder_dim=512, 
                       decoder_depth=6, 
                       decoder_heads=8, 
                       alibi_num_heads=4, 
                       use_abs_pos_emb=False, 
                       cross_attend=True, 
                       alibi_pos_bias=True, 
                       rotary_xpos=True, 
                       attn_flash=True, 
                       qk_norm=True)

# Assume we have an image tensor 'img' of size (1, 256, 256, 3) and 
# a text tensor 'text' of size (1, 50)

# Run the model
output = model(img, text)
```

This will encode `img` using the `ViTransformerWrapper` and then use the encoded embeddings as the context for the `Transformer` to generate a sequence of tokens from `text`. The sequence of tokens, `output`, is the result.
