# PalmE Class Documentation

This documentation covers the `PalmE` class of the `zeta.models` module. This class inherits from PyTorch's `torch.nn.Module` base class for all neural network modules. It's the starting point for creating models in PyTorch; such models can include layers which in turn can also be modules themselves..

The `PalmE` class implements an encoder-decoder architecture useful for solving a variety of tasks by having the encoder extract information from input data which the decoder then uses to generate outputs.

## Class Definition 

The `PalmE` class is constructed as follows:

```python
class PalmE(torch.nn.Module):
    def __init__(
        self,
        image_size=256,
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
        qk_norm=True,
    ):
```

### Parameters 

| Parameter | Type | Description |
| --- | --- | --- |
| `image_size` | int | Size of the input images. Default value is 256. |
| `patch_size` | int | Size of the patches to divide input images into. Default value is 32. |
| `encoder_dim` | int | Dimensionality of the encoder. Default value is 512. |
| `encoder_depth` | int | Number of layers in the encoder. Default value is 6. |
| `encoder_heads` | int | Number of attention heads in the encoder. Default value is 8. |
| `num_tokens` | int | Number of tokens in the input text. Default value is 20000. |
| `max_seq_len` | int | Maximum length of text sequences. Default value is 1024. |
| `decoder_dim` | int | Dimensionality of the decoder. Default value is 512. |
| `decoder_depth` | int | Number of layers in the decoder. Default value is 6. |
| `decoder_heads` | int | Number of attention heads in the decoder. Default value is 8. |
| `alibi_num_heads` | int | Number of heads for the alibi attention mechanism in the decoder. Default value is 4. |
| `use_abs_pos_emb` | bool | Whether to use absolute positional encoding in the decoder. Default is False. |
| `cross_attend` | bool | Whether the decoder should attend to the encoded image features. Default is True. |
| `alibi_pos_bias` | bool | Whether to use a bias in the alibi attention mechanism. Default is True. |
| `rotary_xpos` | bool | Whether to use the rotary positional encoding in place of the token positional encoding. Default is True. |
| `attn_flash` | bool | Whether to use attention flash in the decoder. Default is True. |
| `qk_norm` | bool | Whether to normalize query and key in the decoder self-attention. Default is True. |

## Methods 

### `__init__()` 

The `__init__()` method initializes the `PalmE` instance, sets up the encoder and decoder, and wraps the decoder in an `AutoregressiveWrapper`.

### `forward()`

The `forward()` method performs forward propagation through the model by using the encoder to generate encoded representations of the input images, and then passing these representations and the input text to the decoder in order to generate the model's outputs. A high level pseudo code example can be:

```python
def forward(self, img, text):
    try:
        encoded = self.encoder(img, return_embeddings=True)
        return self.decoder(text, context=encoded)
    except Exception as error:
        print(f"Failed in forward method: {error}")
        raise
```

## Examples

Below you'll find various examples on how to use the `PalmE` class.

### Example 1: Creating a `PalmE` Instance

Here’s an example of how to instantiate the `PalmE` class with the default parameters:

```python
import torch
from zeta.models import PalmE

model = PalmE()
```
### Example 2: Pass input through the model

In this example, we create random image batch and text batch data, and pass them through our `PalmE` model:

```python
img = torch.rand(16, 3, 256, 256) # batch of 16 images
text = torch.randint(0, 20000, (50, 16)) # batch of 50 token sequences for 16 samples

model = PalmE()
out = model(img, text)
```

### Example 3: Modifying model configuration

Let's modify the model's configuration parameters at instantiation:

```python
model = PalmE(encoder_dim=1024, 
              encoder_depth=8, 
              decoder_dim=1024,
              decoder_depth=8,
              attn_flash=False)
```

Here we modified the `encoder_dim`, `encoder_depth`, `decoder_dim`, `decoder_depth` and `attn_flash` parameters.

## Additional Notes

- The input images should have dimensions `(batch_size, channels, height, width)`. The number of channels should usually be 3 (for RGB images), and the height and width should match the `image_size` parameter. 

- The decoder's parameters can be tuned to balance between computational efficiency and the model's performance on your specific task. 

- The `forward()` method may raise an exception if there's a bad input or a compatibility issue between the inputs' and the model's dimensions. Always make sure to match the dimensions. 

- Please refer to the [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) documentation for general information on PyTorch modules. 

- The `rotary_xpos` feature refers to the rotary positional encoding introduced in the paper [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050). It's an alternative to traditional token positional encodings, and often works better. 

- Always make sure your input tensor types (CPU tensor, CUDA tensor etc.) match the configuration of the model. 

- The `PalmE` class supports the standard PyTorch methods for moving the model to a device (`to(device)`) and setting it to train or eval mode (`train() / eval()`).
