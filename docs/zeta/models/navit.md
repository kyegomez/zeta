# Module/Function Name: NaViT

```python
class NaViT(nn.Module)
```
The `NaViT` class is a subclass of PyTorch's `nn.Module` class. It is a reference architecture for creating multi-layer transformers with a pluggable attention, positional encoding, and optional token dropping.

## Initialization: 

To create a `NaViT` instance, the following parameters need to be specified:

```python
def __init__(
    self,
    *,
    image_size,
    patch_size,
    num_classes,
    dim,
    depth,
    heads,
    mlp_dim,
    channels=3,
    dim_head=64,
    dropout=0.0,
    emb_dropout=0.0,
    token_dropout_prob=None,
)
```

|  Parameter          |     Data Type  | Description                                                             |
|----------------------------|------|-------------------------------------------------------------------------------------------------- |
| image_size                      | int  | The size of the input image.                                                                       |
| patch_size                      | int  | The size of the patch that the model will use for feature representation.                                        |
| num_classes                 | int  | The number of classes in the problem, i.e., the size of the output layer of the model.                      |
| dim                           | int   | Dimension of the model.                                                                         |
| depth                         | int   | The number of transformer layers.                    |
| heads                         | int   | The number of attention heads in the transformer.                    |
| mlp_dim                      | int   | The dimension of the multilayer perceptron in the feedforward network.                         |
| channels                      | int  | The number of input channels. Defaults to 3.      |
| dim_head                     | int   | The dimension of the attention head. Defaults to 64. |
| dropout                       | float | Standard dropout. Defaults to 0. The probability of a feature being zeroed out during training. |
| emb_dropout                | float | Dropout applied to the learned embedding at the beginning of the transformer stack. Defaults to 0. |
| token_dropout_prob     | scalar | The probability of dropping out tokens before the transformer. Optional.|

## `forward` pass: 

The forward method specifies the behavior of the model during its forward pass. It takes an image batch as input and returns the output of the model, which is the class probabilities for each input image. 

```python
def forward(self, batched_images: Union[List[Tensor], List[List[Tensor]]], group_images=False, group_max_seq_len=2048)
```

|  Parameter          |     Data Type            | Description                                                  |
|----------------------------|-----------------|----------------------------------------------------- |
| batched_images           | Tensor or List of Tensors  | The input batch of images.                       |
| group_images             | bool | Whether or not to automatically group the images by maximum sequence length. Default: False. |
| group_max_seq_len   | int | The group maximum sequence length for auto-packing. Default: 2048. |

It outputs a 2D tensor with dimensions `(batch size, number of classes)`, representing the class probabilities for each input image.

## Code example:

```python
import torch
from zeta.models import NaViT

# initialize the model
model = NaViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 1024,
)

# random tensor representing a batch of 10 images, with 3 color channels, each 32x32 pixels
x = torch.randn(10, 3, 32, 32)

# the forward function returns the output of the model, which represents class probabilities for each image.
output = model.forward(x)
print(output.shape)  # prints: torch.Size([10, 10])
```

This example demonstrates how to initialize the NaViT model with a set of parameters, how to represent a batch of images as a tensor, and how to feed the image tensor to the model to get the output. 

The output is a batch of logits tensors where each tensor corresponds to class probabilities of the image. The size of each tensor is equal to the `num_classes`, i.e., every batch of images returns a tensor of dimensions `(batch size, num_classes)`. 

This allows direct comparison with the target labels to compute the loss and to derive the gradients during model training.
