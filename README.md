
![Zeta banner](images/zeta.png)

<p>
  <a href="https://github.com/kyegomez/zeta/blob/main/LICENSE">
    <img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" />
  </a>
  <a href="https://pypi.org/project/zetascale">
    <img alt="PyPI" src="https://badge.fury.io/py/zetascale.svg" />
  </a>
  <a href="https://zeta.readthedocs.io">
    <img alt="Docs" src="https://readthedocs.org/projects/zeta/badge/" />
  </a>
</p>

**Zeta** is a modular PyTorch framework designed to simplify the development of AI models by providing reusable, high-performance building blocks. Think of it as a collection of LEGO blocks for AI each component is carefully crafted, tested, and optimized, allowing you to quickly assemble state-of-the-art models without reinventing the wheel.


<p>
  <a href="https://discord.gg/EamjgSaEQf">
    <img src="https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Join our Discord" />
  </a>
  <a href="https://www.youtube.com/@kyegomez3242">
    <img src="https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white" alt="Subscribe on YouTube" />
  </a>
  <a href="https://www.linkedin.com/in/kye-g-38759a207/">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white" alt="Connect on LinkedIn" />
  </a>
  <a href="https://x.com/kyegomezb">
    <img src="https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white" alt="Follow on X.com" />
  </a>
</p>

## Overview

Zeta provides a comprehensive library of modular components commonly used in modern AI architectures, including:

- **Attention Mechanisms**: Multi-query attention, sigmoid attention, flash attention, and more
- **Mixture of Experts (MoE)**: Efficient expert routing and gating mechanisms
- **Neural Network Modules**: Feedforward networks, activation functions, normalization layers
- **Quantization**: BitLinear, dynamic quantization, and other optimization techniques
- **Architectures**: Transformers, encoders, decoders, vision transformers, and complete model implementations
- **Training Utilities**: Optimization algorithms, logging, and performance monitoring


Each component is designed to be:
- **Modular**: Drop-in replacements that work seamlessly with PyTorch
- **High-Performance**: Optimized implementations with fused kernels where applicable
- **Well-Tested**: Comprehensive test coverage ensuring reliability
- **Production-Ready**: Used in hundreds of models across various domains

## Installation

```bash
pip3 install -U zetascale
```

## Quick Start

### Multi-Query Attention

Multi-query attention reduces memory usage while maintaining model quality by sharing key and value projections across attention heads.

```python
import torch
from zeta import MultiQueryAttention

# Initialize the model
model = MultiQueryAttention(
    dim=512,
    heads=8,
)

# Forward pass
text = torch.randn(2, 4, 512)
output, _, _ = model(text)
print(output.shape)  # torch.Size([2, 4, 512])
```

### SwiGLU Activation

The SwiGLU activation function applies a gating mechanism to selectively pass information through the network.

```python
import torch
from zeta.nn import SwiGLUStacked

x = torch.randn(5, 10)
swiglu = SwiGLUStacked(10, 20)
output = swiglu(x)
print(output.shape)  # torch.Size([5, 20])
```

### Relative Position Bias

Relative position bias quantizes the distance between positions into buckets and uses embeddings to provide position-aware attention biases.

```python
import torch
from torch import nn
from zeta.nn import RelativePositionBias

# Initialize the module
rel_pos_bias = RelativePositionBias()

# Compute bias for attention mechanism
bias_matrix = rel_pos_bias(1, 10, 10)

# Use in custom attention
class CustomAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.rel_pos_bias = RelativePositionBias()

    def forward(self, queries, keys):
        bias = self.rel_pos_bias(queries.size(0), queries.size(1), keys.size(1))
        # Use bias in attention computation
        return None
```

### FeedForward Network

A flexible feedforward module with optional GLU activation and LayerNorm, commonly used in transformer architectures.

```python
import torch
from zeta.nn import FeedForward

model = FeedForward(256, 512, glu=True, post_act_ln=True, dropout=0.2)
x = torch.randn(1, 256)
output = model(x)
print(output.shape)  # torch.Size([1, 512])
```

### BitLinear Quantization

BitLinear performs linear transformation with quantization and dequantization, reducing memory usage while maintaining performance. Based on [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453).

```python
import torch
from torch import nn
import zeta.quant as qt

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = qt.BitLinear(10, 20)

    def forward(self, x):
        return self.linear(x)

model = MyModel()
input = torch.randn(128, 10)
output = model(input)
print(output.size())  # torch.Size([128, 20])
```

### PalmE: Multi-Modal Architecture

A complete implementation of the PalmE multi-modal model architecture, combining a ViT image encoder with a transformer decoder for vision-language tasks.

```python
import torch
from zeta.structs import (
    AutoRegressiveWrapper,
    Decoder,
    Encoder,
    Transformer,
    ViTransformerWrapper,
)

class PalmE(torch.nn.Module):
    """
    PalmE is a transformer architecture that uses a ViT encoder and a transformer decoder.
    
    This implementation demonstrates how to combine Zeta's modular components to build
    a complete multi-modal model architecture.
    """
    
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
        attn_kv_heads=2,
        use_abs_pos_emb=False,
        cross_attend=True,
        alibi_pos_bias=True,
        rotary_xpos=True,
        attn_flash=True,
        qk_norm=True,
    ):
        super().__init__()
        
        # Vision encoder
        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=encoder_dim, 
                depth=encoder_depth, 
                heads=encoder_heads
            ),
        )
        
        # Language decoder
        self.decoder = Transformer(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            use_abs_pos_emb=use_abs_pos_emb,
            attn_layers=Decoder(
                dim=decoder_dim,
                depth=decoder_depth,
                heads=decoder_heads,
                cross_attend=cross_attend,
                alibi_pos_bias=alibi_pos_bias,
                alibi_num_heads=alibi_num_heads,
                rotary_xpos=rotary_xpos,
                attn_kv_heads=attn_kv_heads,
                attn_flash=attn_flash,
                qk_norm=qk_norm,
            ),
        )
        
        # Enable autoregressive generation
        self.decoder = AutoRegressiveWrapper(self.decoder)
    
    def forward(self, img: torch.Tensor, text: torch.Tensor):
        """Forward pass of the model."""
        encoded = self.encoder(img, return_embeddings=True)
        return self.decoder(text, context=encoded)

# Usage
img = torch.randn(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 1024))
model = PalmE()
output = model(img, text)
print(output.shape)
```

### U-Net Architecture

A complete U-Net implementation for image segmentation and generative tasks.

```python
import torch
from zeta.nn import Unet

model = Unet(n_channels=1, n_classes=2)
x = torch.randn(1, 1, 572, 572)
y = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
```

### Vision Embeddings

Convert images into patch embeddings suitable for transformer-based vision models.

```python
import torch
from zeta.nn import VisionEmbedding

vision_embedding = VisionEmbedding(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    contain_mask_token=True,
    prepend_cls_token=True,
)

input_image = torch.rand(1, 3, 224, 224)
output = vision_embedding(input_image)
print(output.shape)
```

### Dynamic Quantization with Niva

Niva provides dynamic quantization for specific layer types, ideal for models with variable runtime activations.

```python
import torch
from torch import nn
from zeta import niva

# Load a pre-trained model
model = YourModelClass()

# Quantize the model dynamically
niva(
    model=model,
    model_path="path_to_pretrained_weights.pt",
    output_path="quantized_model.pt",
    quant_type="dynamic",
    quantize_layers=[nn.Linear, nn.Conv2d],
    dtype=torch.qint8,
)
```

### Fused Operations

Zeta includes several fused operations that combine multiple operations into single kernels for improved performance.

#### FusedDenseGELUDense

Fuses two dense operations with GELU activation for up to 2x speedup.

```python
import torch
from zeta.nn import FusedDenseGELUDense

x = torch.randn(1, 512)
model = FusedDenseGELUDense(512, 1024)
out = model(x)
print(out.shape)  # torch.Size([1, 1024])
```

#### FusedDropoutLayerNorm

Fuses dropout and layer normalization for faster feedforward networks.

```python
import torch
from zeta.nn import FusedDropoutLayerNorm

model = FusedDropoutLayerNorm(dim=512)
x = torch.randn(1, 512)
output = model(x)
print(output.shape)  # torch.Size([1, 512])
```

### Mamba: State Space Model

PyTorch implementation of the Mamba state space model architecture.

```python
import torch
from zeta.nn import MambaBlock

block = MambaBlock(dim=64, depth=1)
x = torch.randn(1, 10, 64)
y = block(x)
print(y.shape)  # torch.Size([1, 10, 64])
```

### FiLM: Feature-wise Linear Modulation

Feature-wise Linear Modulation for conditional feature transformation.

```python
import torch
from zeta.nn import Film

film_layer = Film(dim=128, hidden_dim=64, expanse_ratio=4)
conditions = torch.randn(10, 128)
hiddens = torch.randn(10, 1, 128)
modulated_features = film_layer(conditions, hiddens)
print(modulated_features.shape)  # torch.Size([10, 1, 128])
```

### Model Optimization

The `hyper_optimize` decorator` provides a unified interface for multiple optimization techniques.

```python
import torch
from zeta.nn import hyper_optimize

@hyper_optimize(
    torch_fx=False,
    torch_script=False,
    torch_compile=True,
    quantize=True,
    mixed_precision=True,
    enable_metrics=True,
)
def model(x):
    return x @ x

out = model(torch.randn(1, 3, 32, 32))
print(out)
```

### Direct Policy Optimization (DPO)

DPO implementation for reinforcement learning from human feedback (RLHF) applications.

```python
import torch
from torch import nn
from zeta.rl import DPO

class PolicyModel(nn.Module):
    def __init__(self, dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)

dim = 10
output_dim = 5
policy_model = PolicyModel(dim, output_dim)
dpo_model = DPO(model=policy_model, beta=0.1)

preferred_seq = torch.randint(0, output_dim, (3, dim))
unpreferred_seq = torch.randint(0, output_dim, (3, dim))
loss = dpo_model(preferred_seq, unpreferred_seq)
print(loss)
```

### PyTorch Model Logging

A decorator for comprehensive model execution logging, including parameters, gradients, and memory usage.

```python
import torch
from torch import nn
from zeta.utils.verbose_execution import verbose_execution

@verbose_execution(log_params=True, log_gradients=True, log_memory=True)
class YourPyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 222 * 222, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = YourPyTorchModel()
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)

# Gradient information requires backward pass
loss = output.sum()
loss.backward()
```

### Sigmoid Attention

An attention mechanism that replaces softmax with sigmoid, providing up to 18% speedup while maintaining performance.

```python
import torch
from zeta import SigmoidAttention

batch_size = 32
seq_len = 128
dim = 512
heads = 8

x = torch.rand(batch_size, seq_len, dim)
mask = torch.ones(batch_size, seq_len, seq_len)

sigmoid_attn = SigmoidAttention(dim, heads, seq_len)
output = sigmoid_attn(x, mask)
print(output.shape)  # torch.Size([32, 128, 512])
```

## Documentation

Comprehensive documentation is available at [zeta.apac.ai](https://zeta.apac.ai/).

## Running Tests

Install the pre-commit hooks to run linters, type checking, and a subset of tests on every commit:

```bash
pre-commit install
```

To run the full test suite:

```bash
python3 -m pip install -e '.[testing]'  # Install extra dependencies for testing
python3 -m pytest tests/                # Run the entire test suite
```

For more details, refer to the CI workflow configuration.

## Community

Join our growing community for real-time support, ideas, and discussions on building better AI models.

| Platform    | Link                                                                         | Description                 |
|-------------|------------------------------------------------------------------------------|-----------------------------|
| Docs        | [zeta.apac.ai](https://zeta.apac.ai)                                         | Official documentation      |
| Discord     | [Join our Discord](https://discord.gg/EamjgSaEQf)                            | Live chat & community       |
| Twitter     | [@kyegomez](https://twitter.com/kyegomez)                                    | Follow for updates          |
| LinkedIn    | [The Swarm Corporation](https://www.linkedin.com/company/the-swarm-corporation) | Connect professionally      |
| YouTube     | [YouTube Channel](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ)  | Watch our videos            |

## Contributing

Zeta is an open-source project, and contributions are welcome! If you want to create new features, fix bugs, or improve the infrastructure, we'd love to have you contribute.

**Getting Started:**

- Pick any issue with the `good first issue` tag to get started
- Read our [Contributing Guidelines](CONTRIBUTING.md)
- Check out our [contributing board](https://github.com/users/kyegomez/projects/1) for roadmap discussions

**Report Issues:**

- [Bug Report](https://github.com/kyegomez/zeta/issues/new/choose)
- [Feature Request](https://github.com/kyegomez/zeta/issues/new/choose)

## Our Contributors

Thank you to all of our contributors who have built this great framework 🙌

<a href="https://github.com/kyegomez/zeta/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kyegomez/zeta" alt="Contributors" />
</a>

---


## Citation

If you use Zeta in your research or projects, please cite it:

```bibtex
@misc{zetascale,
    title = {Zetascale Framework},
    author = {Kye Gomez},
    year = {2024},
    howpublished = {\url{https://github.com/kyegomez/zeta}},
}
```

## License

Apache 2.0 License
