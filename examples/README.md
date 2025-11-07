# Examples Overview

This directory contains comprehensive examples demonstrating various features and use cases of the Zeta library. The examples are organized by category to help you quickly find relevant implementations.

## Table of Contents

- [Models](#models)
- [Modules](#modules)
- [Training](#training)
- [Operations](#operations)
- [Structures](#structures)
- [Tokenizers](#tokenizers)
- [Cython Tests](#cython-tests)
- [Todo](#todo)

## Models

Examples demonstrating different model architectures and implementations.

### Core Models

- **`simple_transformer.py`** - A basic transformer implementation showcasing the fundamental components including shaped attention, feedforward layers, and residual connections. Demonstrates how to build a simple transformer from scratch using Zeta's building blocks.

- **`gpt4.py`** - Implementation of a GPT-4 style language model using Zeta's transformer components.

- **`gpt4_multimodal.py`** - Multimodal extension of GPT-4 that can process both text and visual inputs.

- **`cobra.py`** - Implementation of the Cobra model architecture.

- **`nirvana.py`** - Nirvana model implementation demonstrating advanced transformer architectures.

- **`toka_master_gpt.py`** - Toka Master GPT model implementation.

### Specialized Models

- **`agi/model.py`** - Artificial General Intelligence model implementation exploring advanced architectures.

- **`videos/spectra.py`** - Video processing model using spectral analysis techniques.

- **`transformer_real_time_learning/transformer_moe_liquid_real_time.py`** - Real-time learning transformer with Mixture of Experts (MoE) architecture and liquid neural network concepts.

- **`evo_transformer_mutate/`** - Evolutionary transformer model that simulates reproduction and evolution using evolutionary algorithms. Features Multi-Query Attention (MQA) and Mixture of Experts (MoE) architecture. Includes detailed documentation in the README.md file.

## Modules

Examples of individual neural network modules and components.

- **`flash_attention.py`** - Demonstration of Flash Attention implementation, showing how to use the optimized attention mechanism for improved performance and memory efficiency.

- **`cross_attend.py`** - Cross-attention module example for processing relationships between different input sequences.

- **`sigmoid_attn.py`** - Sigmoid-based attention mechanism implementation.

- **`fractoral_norm.py`** - Fractal normalization technique example.

- **`viusal_expert_example.py`** - Visual expert module demonstrating specialized processing for visual inputs.

- **`flow_matching_modules/`** - Flow matching implementations:
  - **`flow_matching.py`** - Basic flow matching module
  - **`flow_moe.py`** - Flow matching with Mixture of Experts architecture

## Training

Comprehensive training examples covering various domains and techniques.

### General Training

- **`fsdp.py`** - Fully Sharded Data Parallel (FSDP) training example for distributed training across multiple GPUs.

- **`muon.py`** - Example demonstrating the Muon optimizer usage, showing how to combine different optimizers for different parameter groups.

- **`new_optimizer/fa_optimizer.py`** - Example of a custom optimizer implementation.

### Domain-Specific Training

- **`face_recog/`** - Face recognition training examples:
  - **`face_vit.py`** - Vision Transformer for face recognition
  - **`frt.py`** - Face recognition transformer
  - **`rf_prediction.py`** - Random forest prediction for face recognition

- **`food/food_detect.py`** - Food detection model training example.

- **`stock_prediction/`** - Stock market prediction examples:
  - **`ts.py`** - Time series stock prediction
  - **`ts_real_time.py`** - Real-time stock prediction
  - **`ts_stock_model_realtime.py`** - Real-time stock model implementation
  - Includes data files: `energy_stocks_analysis.xlsx`, `energy_stocks_predictions.csv`, `scaler.joblib`

- **`weather_training/v_w_transformer.py`** - Weather prediction using Vision-Weather transformer architecture.

- **`earth_quake/tea.py`** - Earthquake prediction model training.

- **`protein_g/protein_gen_transformer.py`** - Protein generation using transformer models.

- **`radio_frequency/`** - Radio frequency signal processing:
  - **`rf_model.py`** - RF model implementation
  - **`rf_model.joblib`** - Trained model checkpoint

- **`visual_reasoning/vit_siglip.py`** - Visual reasoning using Vision Transformer with SigLIP architecture.

- **`yolo_alt/model.py`** - Alternative YOLO object detection model implementation.

### Advanced Training Techniques

- **`gan/`** - Generative Adversarial Network examples:
  - **`gan.py`** - Basic GAN implementation
  - **`new_gan.py`** - Enhanced GAN with improved architecture

- **`mo/`** - Mamba Omega training:
  - **`mamba_omega.py`** - Mamba Omega model implementation
  - **`train.py`** - Training script
  - **`requirements.txt`** - Dependencies

- **`evo_t/`** - Evolutionary transformer training:
  - **`transformers_evolutionary_train.py`** - Main training script for evolutionary transformer models
  - Contains training logs and evolved model checkpoints

- **`vis/`** - Vision model training examples:
  - **`transformer.py`** - Vision transformer
  - **`model.py`** - General vision model
  - **`t2.py`, `t3.py`, `t4.py`, `t5.py`** - Various vision model variants

## Operations

Low-level operations and utilities.

- **`laplace.py`** - Laplace transform operations example.

## Structures

High-level model structures and architectures.

- **`transformer.py`** - Complete transformer structure example demonstrating how to use Zeta's Transformer and Decoder components to build a full model.

## Tokenizers

Tokenization examples and implementations.

- **`token_monster.py`** - Token Monster tokenizer implementation example, demonstrating advanced tokenization techniques.

## Cython Tests

Cython extension examples for performance-critical operations.

- **`mqa.pyx`** - Multi-Query Attention Cython implementation
- **`mqa_test.py`** - Tests for the MQA Cython extension
- **`new_c_example.py`** - C extension example
- **`torch_extension.pyx`** - PyTorch Cython extension example
- **`setup.py`** - Setup script for building Cython extensions

## Todo

Experimental or work-in-progress implementations.

- **`dit_block.py`** - Diffusion Transformer block implementation (work in progress)
- **`hyper_attention.py`** - Hyper attention mechanism (experimental)
- **`multi_head_latent_attention.py`** - Multi-head latent attention implementation (experimental)

## Getting Started

To run any example, navigate to the specific directory and execute the Python file:

```bash
cd examples/models
python simple_transformer.py
```

Most examples are self-contained and can be run directly. Some training examples may require additional data files or configuration. Refer to individual example files for specific requirements and usage instructions.

## Notes

- Examples are provided for educational and reference purposes
- Some examples may require additional dependencies beyond the base Zeta installation
- Training examples may require significant computational resources
- Experimental examples in the `todo/` directory are subject to change
