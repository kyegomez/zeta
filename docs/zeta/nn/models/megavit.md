# `MegaVit`: A Vision Transformer Model

MegaVit is a variant of the Vision Transformer (ViT) model that has been designed for computer vision tasks. It represents a significant advancement in the application of transformers to visual data, offering a scalable and effective means of processing image data without relying on conventional convolutional networks.

## Overview

Vision Transformers represent an innovative approach to processing visual data. Instead of relying on convolutional layers to extract features from images, they use the power of transformers to analyze patches of an image. Each patch is treated as a token, similar to how natural language processing models treat words as tokens.

The MegaVit model uses the transformer architecture to process these image patches, aggregating information across the image to produce an output representation that can be used for classification or other tasks.

### Key Concepts

1. **Patch Embedding**: The image is split into fixed-size patches. Each patch is linearly embedded into a vector. These vectors serve as the input tokens for the transformer.
2. **Positional Embedding**: Due to the lack of inherent sequence in images, a positional embedding is added to provide the model with information about the location of each patch.
3. **Transformer Layers**: These are the core of the model, processing the patch embeddings through multiple layers of attention and feed-forward networks.
4. **Pooling**: After processing through the transformers, the patch representations are pooled (either by taking the class token or averaging) to produce a singular representation of the image.
5. **MLP Head**: A final multi-layer perceptron (MLP) head is used to produce the output, such as class probabilities for an image classification task.

## Model Definition

### `MegaVit`

```python
class MegaVit(nn.Module):
```

#### Parameters:

- `image_size` (`int`): The size of the input image. Both height and width must be divisible by the `patch_size`.
- `patch_size` (`int`): The size of each patch. The image is divided into patches of this size.
- `num_classes` (`int`): The number of output classes for classification.
- `dim` (`int`): The dimensionality of the patch embedding.
- `depth` (`int`): The number of transformer layers.
- `heads` (`int`): The number of attention heads in the multi-head attention mechanism.
- `mlp_dim` (`int`): The dimensionality of the MLP layers within the transformer.
- `pool` (`str`): The pooling method to use. Can be either 'cls' for class token pooling or 'mean' for mean pooling.
- `channels` (`int`): The number of input channels (e.g., 3 for RGB images).
- `dim_head` (`int`): The dimensionality of each head in the multi-head attention mechanism.
- `dropout` (`float`): Dropout rate applied within the transformer.
- `emb_dropout` (`float`): Dropout rate applied to the embeddings.

#### Returns:

- `torch.Tensor`: A tensor of shape `(batch_size, num_classes)`, representing the class logits for each image in the batch.

#### Methods:

##### `forward(img: torch.Tensor) -> torch.Tensor`

- Parameters:
  - `img` (`torch.Tensor`): A batch of images of shape `(batch_size, channels, image_size, image_size)`.
- Returns:
  - `torch.Tensor`: The class logits for each image in the batch.

#### Architecture:

1. **Patch Embedding**: The input image is passed through a patch embedding layer, converting the image patches into linear embeddings.
2. **Positional Embedding**: Adds positional information to the patch embeddings.
3. **Dropout**: Dropout is applied for regularization.
4. **Transformer**: The embedded patches are processed through a series of transformer layers.
5. **Pooling**: The patch embeddings are pooled to produce a single vector representation for the entire image.
6. **MLP Head**: An MLP head produces the final output logits for classification.
7. **Output**: The logits are returned.

### Usage Example:

```python
from zeta.models import MegaVit

model = MegaVit(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)
preds = model(img) # Shape: (1, 1000)
```

## Notes:

1. The image dimensions must be divisible by the patch size. This ensures that the image can be evenly divided into patches.
2. The choice of pooling ('cls' vs 'mean') can have an impact on performance. 'cls' pooling uses a special class token, while 'mean' pooling averages the patch embeddings.
3. Regularization through dropout can help prevent overfitting, especially when training on smaller datasets.
