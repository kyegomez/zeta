# Module Name: MegaVit

The MegaVit is a class in Python that implements the model from the paper [When Vision Transformers Outperform CNNs](https://arxiv.org/abs/2106.14759). 

## Introduction

The class implements a vision transformer model that can provide state-of-the-art performance in computer vision tasks when compared to traditional convolutional neural networks (CNNs). The vision transformer model treats an image as a sequence of one-dimensional patches and applies the transformer model on these patches. It is initialized with image size, patch size, number of classes, embedding dimension, depth of transformer model, number of heads for the multi-head attention mechanism, dimension of multi-layer perceptron (MLP), type of pooling method, and dropout rates.

## Class Definition

```python
class MegaVit(nn.Module):
```

This class inherits from `nn.Module`, which is the base class for all neural network modules in Pytorch.

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
    pool="cls",
    channels=3,
    dim_head=64,
    dropout=0.0,
    emb_dropout=0.0,
):
```

The initialization function for the `MegaVit` class. This function initializes various parameters and layers of the model.

- `image_size`: Size of the input image. It should be an integer. This is an input argument to the `MegaVit` initializer.
- `patch_size`: Size of the patches into which the input image is divided. It should be an integer.
- `num_classes`: Number of output classes. It should be an integer.
- `dim`: It is the dimension of the embeddings.
- `depth`: This integer represents the depth of the transformer.
- `heads`: This integer indicates the number of heads in the multi-head attention mechanism of the transformer.
- `mlp_dim`: This integer represents the number of dimensions in the MLP layer.
- `pool`: This is a string representing the type of pooling used. It can either be 'cls' or 'mean'.
- `channels`: This integer represents the number of channels in the input image.
- `dim_head`: This integer is the dimension of the transformers head.
- `dropout`: This floating-point number represents the dropout rate.
- `emb_dropout`: This floating-point number is the dropout rate for the embeddings.

```python
def forward(self, img):
```

The forward function defines the forward pass of the network. It receives an input image and generates an output prediction.

- `img`: A Pytorch tensor representing the input image.

## Usage Example

Here is a basic usage example of the `MegaVit` class:

```python
import torch
from torch.nn import Module
from numpy import random
from zeta.models import MegaVit

# Define model hyperparameters
model_hparams = {
    "image_size": 256,
    "patch_size": 32,
    "num_classes": 1000,
    "dim": 512,
    "depth": 6,
    "heads": 8,
    "mlp_dim": 1024,
    "dropout": 0.1,
    "emb_dropout": 0.1,
}

# Initialize MegaVit model
model = MegaVit(**model_hparams)

# Get random image
img = torch.from_numpy(random.rand(1, 3, model_hparams["image_size"], model_hparams["image_size"])).float()

# Get model prediction
preds = model(img)

print(preds)
```

This will output the model's prediction for the input image.

## Reference 

- [When Vision Transformers Outperform CNNs](https://arxiv.org/abs/2106.14759)

This class directly corresponds to the model presented in the above-mentioned paper. Reading this paper may provide additional insights into working and theory of this class. 

## Additional Information

Below is a brief explanation of how the `MegaVit` model works:

1. The input image is passed through the `to_patch_embedding` layer, which first rearranges the image into patches, then applies layer normalization and linear transformation on each patch separately.
2. The positional embeddings are added to these patch embeddings.
3. Dropout is applied as a regularization technique.
4. The transformer is applied to process the patch embeddings.
5. The pooling is applied to the output of the transformer. The type of pooling depends on the `pool` parameter ('cls' or 'mean').
6. The MLP head is applied to obtain prediction for each class.
7. The model returns these predictions.
