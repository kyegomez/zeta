# MaxVit Class Documentation

The `MaxVit` class in the `zeta.models` module is a neural network module for constructing Vision Transformers (ViT) with MixUp functionality. This class extends PyTorch's native `nn.Module` class while adding various features suited for implementing ViTs. The following sections will provide additional details:

## Class Definition

```python
class MaxVit(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head: int = 32,
        dim_conv_stem=None,
        window_size: int = 7,
        mbconv_expansion_rate: int = 4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.01,
        channels=3,
    ):
```

### Parameters
| Parameters            |          Type        |                                                                                                                            Description                                             |
|-----------------------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `num_classes`       | `int`                | The number of classes in the classification task.                                                                                                                                                |
| `dim`               | `int`                | The dimension of the input data.                                                                                                 |
| `depth`             | `list`               | Tuple indicating the number of transformer blocks at a given stage.    |
| `dim_head`           | `int` (Default = 32) |  The dimensionally of the transformer's heads.                                     |
| `dim_conv_stem`    | `int` (Default = None)| The dimensionality of the convolutional stem. If not provided, the dimension of the input is used. |
| `window_size`        | `int` (Default = 7)  | The size of the sliding windows used for efficient grid-like attention. |
| `mbconv_expansion_rate` | `int` (Default = 4) | Expansion rate used in Mobile Inverted Residual Bottleneck (MBConv) used in the `block`. |
| `mbconv_shrinkage_rate` | `float` (Default = 0.25) | Shrinkage rate used in Mobile Inverted Residual Bottleneck (MBConv) used in the `block`. |
| `dropout`            | `float` (Default = 0.01) | The dropout rate for regularization.                         |
| `channels`           | `int`   (Default = 3) | Number of input channels.                                     |

## Functions / Methods

### `forward(x, texts=None, cond_fns=None, cond_drop_prob=0.0, return_embeddings=False)`

This function carries out the forward propagation through the `MaxVit` model given an input `x`. 

#### Parameters 
| Parameter            |          Type        |                                                                                                                            Description                                             |
|-----------------------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `x`                    | `torch.Tensor`       | The input tensor to the `MaxVit` model.                                                                                                     |
| `texts`                |`List[str]` (Optional)| list of textual data for interpreting image data |
| `cond_fns`           |`Tuple[Callable, ...]` (Optional)|  List of conditional functions to apply per layer |
| `cond_drop_prob` |`float` (Default = 0.0) | Conditional dropout probability. |
| `return_embeddings` |`bool` (Default = False) | Whether to return embeddings instead of class scores.|

#### Returns
Returns the output of the multi-layer transformer, which could either be the class scores (default) or embeddings based on `return_embeddings` value.

## Example Usage

```python
from zeta.models import MaxVit

model = MaxVit(num_classes=10, dim=512, depth=(3,2), dim_head=64, channels=3)

x = torch.randn(1, 3, 224, 224)  # suppose we have an random tensor representing an image

out = model(x)  # forward pass

print(out.shape)  # torch.Size([1, 10])
```

## Overview

The `MaxVit` model is essentially a combination of vision transformers and efficient blocks (based on MobileNet family). First, the input passes through a convolutional stem. Afterward, the data flow through several stages. Each stage consists of a sequence of blocks, and each block is a combination of a Mobile Inverted Residual Bottleneck (MBConv) followed by the Transformer layers. Finally, the output to predict the classifications is obtained through the MLP head. 

In addition to the traditional `forward` functionality, `MaxVit` also supports conditional functions that can be used to modify the network behavior per layer, adding a layer of flexibility to the model. Furthermore, the model supports the option to return the transformer embeddings, making it applicable for other tasks beyond simple classification.

## Note:
The forward method of `MaxVit` is beartyped for type checking which enforces strong typing, improving the efficiency of the class.
