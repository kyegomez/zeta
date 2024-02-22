# Module/Class Name: ViT (Vision Transformer)

The Vision Transformer (ViT) is a class designed as part of the `zeta.models` library. It builds upon the efficient Transformer architecture for applying convolutions for image recognition tasks. The ViT class inherits the properties and methods from PyTorch's built-in `torch.nn.Module` class. This class repurposes the Transformer architecture for image processing tasks by dividing the image into numerous patches and feeding them into the Transformer.

## Class Definition

```python
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, attn_layers, channels=3, num_classes=None, post_emb_norm=False, emb_dropout=0.0):
```
This class takes the following parameters as inputs:

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| image_size | int | The dimensions (height and width) of the input image. | - |
| patch_size | int | The dimensions of each image patch to be input to the Transformer. | - |
| attn_layers | `Encoder` | A sequence of attention layers defined using the `Encoder` class. | - |
| channels | int | The number of color-bands (usually RGB). | 3 |
| num_classes | int | The number of classes to be detected, otherwise `None` for unsupervised learning scenarios. | `None` |
| post_emb_norm | bool | Whether to apply layer-normalization to the embeddings. | `False` |
| emb_dropout | float | The probability of an element to be zeroed in dropout. | `0.0` |

## Method Definitions

Here are the core methods of the `ViT` class:

1. `__init__`

This method initializes the instance and sets up the various components of the Transformer, including the positional embeddings, the sequence of attention layers, and the output MLP head.

2. `forward`

This method defines the feedforward computations of the ViT, starting from the division of the input image into patches, the conversion of patches into embeddings, applying attention layers, and, if specified, the MLP head for classification output.

## Usage Examples

Here, we demonstrate how to use the ViT class.

```python
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from zeta.models import Encoder, ViT

# Load an image and apply some pre-processing
img = Image.open("path_to_your_image.jpg")
transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]  # Resize image to 224x224
)
img_tensor = transform(img).unsqueeze(0)

# Define an Encoder with attention layers
encoder = Encoder(dim=512, depth=12)

# Instantiate a ViT model
vit_model = ViT(
    image_size=224,
    patch_size=16,
    attn_layers=encoder,
    channels=3,
    num_classes=1000,
    post_emb_norm=True,
    emb_dropout=0.1,
)

# Generate outputs using the ViT model
outputs = vit_model(img_tensor, return_embeddings=True)

print("Output shape (with embeddings):", outputs.size())

outputs = vit_model(img_tensor, return_embeddings=False)

print("Output shape (without embeddings):", outputs.size())
```

This code presents a usage scenario of the `ViT` class. It illustrates how to load an image, preprocess it, define an `Encoder` instance with attention layers, instantiate a `ViT` model with the defined `Encoder`, and generate outputs (embeddings and class probabilities) using the instantiated `ViT` model.
