# `TransformerBlock` Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Purpose and Functionality](#purpose-and-functionality)
3. [Class: `TransformerBlock`](#class-transformerblock)
   - [Initialization](#initialization)
   - [Parameters](#parameters)
   - [Attention Mechanism](#attention-mechanism)
   - [Multi-Head Attention](#multi-head-attention)
   - [Rotary Embedding](#rotary-embedding)
   - [Feedforward Network](#feedforward-network)
   - [Caching and Optimization](#caching-and-optimization)
4. [Usage Examples](#usage-examples)
   - [Basic Usage](#basic-usage)
   - [Fine-Tuning](#fine-tuning)
5. [Additional Information](#additional-information)
   - [Layernorm](#layernorm)
   - [Position Embeddings](#position-embeddings)
6. [References](#references)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the Zeta documentation for the `TransformerBlock` class! Zeta is a versatile library that offers tools for efficient training of deep learning models using PyTorch. This documentation will provide a comprehensive overview of the `TransformerBlock` class, its architecture, purpose, and usage.

---

## 2. Purpose and Functionality <a name="purpose-and-functionality"></a>

The `TransformerBlock` class is a fundamental component of the Zeta library. It is designed to be used within a transformer-based architecture, and its primary purpose is to process input data efficiently. Below, we'll explore the key functionalities and features of the `TransformerBlock` class.

---

## 3. Class: `TransformerBlock` <a name="class-transformerblock"></a>

The `TransformerBlock` class is the building block of transformer-based models. It performs various operations, including multi-head attention and feedforward network, to process input data. Let's dive into the details of this class.

### Initialization <a name="initialization"></a>

To create a `TransformerBlock` instance, you need to specify various parameters and configurations. Here's an example of how to initialize it:

```python
TransformerBlock(
    dim=512,
    dim_head=64,
    causal=True,
    heads=8,
    qk_rmsnorm=False,
    qk_scale=8,
    ff_mult=4,
    attn_dropout=0.0,
    ff_dropout=0.0,
    use_xpos=True,
    xpos_scale_base=512,
    flash_attn=False
)
```

### Parameters <a name="parameters"></a>

- `dim` (int): The dimension of the input data.

- `dim_head` (int): The dimension of each attention head.

- `causal` (bool): Whether to use a causal (auto-regressive) attention mechanism. Default is `True`.

- `heads` (int): The number of attention heads. 

- `qk_rmsnorm` (bool): Whether to apply root mean square normalization to query and key vectors. Default is `False`.

- `qk_scale` (int): Scaling factor for query and key vectors. Used when `qk_rmsnorm` is `True`. Default is `8`.

- `ff_mult` (int): Multiplier for the feedforward network dimension. Default is `4`.

- `attn_dropout` (float): Dropout probability for attention layers. Default is `0.0`.

- `ff_dropout` (float): Dropout probability for the feedforward network. Default is `0.0`.

- `use_xpos` (bool): Whether to use positional embeddings. Default is `True`.

- `xpos_scale_base` (int): Scaling factor for positional embeddings. Default is `512`.

- `flash_attn` (bool): Whether to use Flash Attention mechanism. Default is `False`.

### Attention Mechanism <a name="attention-mechanism"></a>

The `TransformerBlock` class includes a powerful attention mechanism that allows the model to focus on relevant parts of the input data. It supports both regular and Flash Attention.

### Multi-Head Attention <a name="multi-head-attention"></a>

The class can split the attention mechanism into multiple heads, allowing the model to capture different patterns in the data simultaneously. The number of attention heads is controlled by the `heads` parameter.

### Rotary Embedding <a name="rotary-embedding"></a>

Rotary embeddings are used to enhance the model's ability to handle sequences of different lengths effectively. They are applied to query and key vectors to improve length extrapolation.

### Feedforward Network <a name="feedforward-network"></a>

The `TransformerBlock` class includes a feedforward network that processes the attention output. It can be customized by adjusting the `ff_mult` parameter.

### Caching and Optimization <a name="caching-and-optimization"></a>

The class includes mechanisms for caching causal masks and rotary embeddings, which can improve training efficiency. It also provides options for fine-tuning specific modules within the block.

---

## 4. Usage Examples <a name="usage-examples"></a>

Now, let's explore some usage examples of the `TransformerBlock` class to understand how to use it effectively.

### Basic Usage <a name="basic-usage"></a>

```python
# Create a TransformerBlock instance
transformer_block = TransformerBlock(dim=512, heads=8)

# Process input data
output = transformer_block(input_data)
```

### Fine-Tuning <a name="fine-tuning"></a>

```python
# Create a TransformerBlock instance with fine-tuning modules
lora_q = YourCustomModule()
lora_k = YourCustomModule()
lora_v = YourCustomModule()
lora_o = YourCustomModule()

transformer_block = TransformerBlock(
    dim=512,
    heads=8,
    finetune_modules=(lora_q, lora_k, lora_v, lora_o)
)

# Process input data
output = transformer_block(input_data)
```

---

## 5. Additional Information <a name="additional-information"></a>

### Layernorm <a name="layernorm"></a>

The `TransformerBlock` class uses layer normalization (layernorm) to normalize input data before processing. This helps stabilize and accelerate training.

### Position Embeddings <a name="position-embeddings"></a>

Position embeddings are used to provide the model with information about the position of tokens

 in the input sequence. They are crucial for handling sequences of different lengths effectively.

---

## 6. References <a name="references"></a>

- [Original Transformer Paper](https://arxiv.org/abs/1706.03762)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Flash Attention: Scaling Vision Transformers with Hybrid Attention for Image and Video Recognition](https://arxiv.org/abs/2203.08124)
- [Layer Normalization](https://arxiv.org/abs/1607.06450)

This documentation provides a comprehensive guide to the `TransformerBlock` class in the Zeta library, explaining its purpose, functionality, parameters, and usage. You can now effectively integrate this class into your deep learning models for various natural language processing tasks and beyond.