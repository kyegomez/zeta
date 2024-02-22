# MultiModalMambaBlock

#### Table of Contents
- [Introduction](#introduction)
- [Fusion Method and Model Architecture](#fusion-method-and-model-architecture)
- [Usage and Examples](#usage-and-examples)
- [Further References](#further-references)

<a name="introduction"></a>
## Introduction
The MultiModalMambaBlock is a PyTorch module designed to combine text and image embeddings using a multimodal fusion approach. It provides methods for attention-based fusion using a Mamba block, ViT encoder, and image/text embeddings. By using a variety of fusion methods, the MultiModalMambaBlock aims to facilitate the learning of joint representations from different modalities.

<a name="fusion-method-and-model-architecture"></a>
## Fusion Method and Model Architecture

### Args
| Args            | Description                                                                    |
|-----------------|--------------------------------------------------------------------------------|
| `dim`           | The dimension of the embeddings.                                               |
| `depth`         | The depth of the Mamba block.                                                   |
| `dropout`       | The dropout rate.                                                              |
| `heads`         | The number of attention heads.                                                 |
| `d_state`       | The dimension of the state in the Mamba block.                                 |
| `image_size`    | The size of the input image.                                                   |
| `patch_size`    | The size of the image patches.                                                 |
| `encoder_dim`   | The dimension of the encoder embeddings.                                       |
| `encoder_depth` | The depth of the encoder.                                                      |
| `encoder_heads` | The number of attention heads in the encoder.                                  |
| `fusion_method` | The multimodal fusion method to use. Can be one of ["mlp", "concat", "add"].   |

### Module Architecture
- **Mamba Block:** Implements a transformer-like Mamba block for attention-based fusion of embeddings.
- **ViT Encoder:** Utilizes a Vision Transformer encoder for image-based attention encoding.
- **Fusion Methods:** Provides support for various fusion methods, including MLP fusion, concatenation, addition, and visual expert methods.

<a name="usage-and-examples"></a>
## Usage and Examples

```python
x = torch.randn(1, 16, 64)
y = torch.randn(1, 3, 64, 64)
model = MultiModalMambaBlock(
    dim=64,
    depth=5,
    dropout=0.1,
    heads=4,
    d_state=16,
    image_size=64,
    patch_size=16,
    encoder_dim=64,
    encoder_depth=5,
    encoder_heads=4,
    fusion_method="mlp",
)
out = model(x, y)
print(out.shape)
```

```python
# Checking the current fusion method
model.check_fusion_method()
```

<a name="further-references"></a>
## Further References
For additional information and detailed usage, please refer to the official documentation of the `MultiModalMambaBlock` module.

**Note:** The architecture and methods used in the `MultiModalMambaBlock` module are designed to address the specific challenge of joint attention-based multimodal representation learning. The selected `fusion_method` and fusion approach can significantly impact the model performance, and care should be taken when choosing the appropriate method for a particular use case.
