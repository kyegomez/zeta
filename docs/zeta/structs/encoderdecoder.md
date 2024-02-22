# Module/Class Name: EncoderDecoder

The `EncoderDecoder` class is a module that brings together an encoder and a decoder for sequence-to-sequence tasks. This design helps facilitate the transformation of an input sequence to an output sequence, with each sequence potentially being of a different length. 

Applications of sequence-to-sequence tasks include machine translation, speech recognition, and text summarization.

![Image](https://miro.medium.com/max/1800/1*n-IgHZM5baBUjq0T7RYDBw.gif)

<image caption: Sequence to Sequence model>

This EncoderDecoder class requires an argparse.Namespace object as well as optional Tensor objects for the encoder embed tokens and positions and the decoder embed tokens and positions.

## Class Definition

```python
class EncoderDecoder(nn.Module):
    """
    A module that combines an encoder and a decoder for sequence-to-sequence tasks.

    Args:
        args (argparse.Namespace): The arguments passed to the module.
        encoder_embed_tokens (torch.Tensor, optional): The input embeddings for the encoder. Defaults to None.
        encoder_embed_positions (torch.Tensor, optional): The positions of the encoder input embeddings. Defaults to None.
        decoder_embed_tokens (torch.Tensor, optional): The input embeddings for the decoder. Defaults to None.
        decoder_embed_positions (torch.Tensor, optional): The positions of the decoder input embeddings. Defaults to None.
        output_projection (torch.Tensor, optional): The projection layer for the decoder output. Defaults to None.
        **kwargs: Additional keyword arguments.

    Attributes:
        args (argparse.Namespace): The arguments passed to the module.
        encoder (Encoder): The encoder module.
        decoder (Decoder): The decoder module.
    """


...
```

This class has two major attributes: `encoder` and `decoder`. These attributes store the encoder and decoder modules used in sequence-to-sequence tasks.

## Initialization of EncoderDecoder

The `EncoderDecoder` class is initialized as follows:

```python
def __init__(
    self,
    args,
    encoder_embed_tokens=None,
    encoder_embed_positions=None,
    decoder_embed_tokens=None,
    decoder_embed_positions=None,
    output_projection=None,
    **kwargs,
):
```

## Init Parameters
The EncoderDecoder class takes the following parameters during its initialization:

| Parameter| Type | Description |
|---|---|---|
|args| argparse.Namespace| The namespace containing all the arguments needed to initialize the module.|
|encoder_embed_tokens|torch.Tensor (optional)| The input embeddings for the encoder.|
|encoder_embed_positions| torch.Tensor (optional)| The position indices for the encoder input embeddings.|
|decoder_embed_tokens|torch.Tensor (optional)| The input embeddings for the decoder.|
|decoder_embed_positions| torch.Tensor (optional)| The position indices for the decoder input embeddings.|
|output_projection| torch.Tensor (optional)| The projection matrix for the decoder output.|
|**kwargs|dict| A dictionary of additional keyword arguments.|


During initialization, the `EncoderDecoder` class checks if all embeddings should be shared between the encoder and decoder. If not, it initializes the encoder and decoder with their respective embed tokens and position indices.


## Forward Method Definition

```python
def forward(
    self,
    src_tokens,
    prev_output_tokens,
    return_all_hiddens=False,
    features_only=False,
    **kwargs,
):
```
This method executes the forward pass of the module.

## Forward Method Parameters
| Parameter| Type | Description |
|---|---|---|
|src_tokens|torch.Tensor| The source tokens.|
|prev_output_tokens|torch.Tensor| The previous output tokens.|
|return_all_hiddens|bool (optional)| Whether to return all hidden states. Default is `False`.|
|features_only| bool (optional)| Whether to return only the features. Default is `False`.|
|**kwargs|dict| A dictionary of additional keyword arguments.|


## Usage Example:

```python
# Imports
import torch

from zeta.structs import Decoder, Encoder, EncoderDecoder

# Arguments
args = argparse.Namespace(share_all_embeddings=True)
src_tokens = torch.tensor([1, 2, 3])
prev_output_tokens = torch.tensor([0, 1, 2])

# Define EncoderDecoder
enc_dec = EncoderDecoder(args)

# Forward Pass
decoder_out = enc_dec(src_tokens, prev_output_tokens)
```
This returns the output of the decoder module. 

## Note:

- `Encoder` and `Decoder` are assumed to be modules input to the `EncoderDecoder` class.
- Ensure that your input tensors are of the right shape and type (LongTensor for token indices and FloatTensor for embedding vectors).
- When training a model using the `EncoderDecoder` class, make sure to use the appropriate loss function that matches your specific task (e.g., CrossEntropyLoss for classification tasks).
- The argparse.Namespace class is used to hold the arguments needed by the module. It's a simple class that allows access to undefined attributes.
