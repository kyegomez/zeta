# Zeta - The Forerunner Library for Transforming AI

<p>
  <a href="https://github.com/kygomez/zeta/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://pypi.org/project/zeta"><img alt="MIT License" src="https://badge.fury.io/py/zeta.svg" /></a>
</p>

Zeta is a PyTorch library that empowers researchers and developers to amplify Transformers to new levels of efficiency and effectiveness. It implements pioneering research to augment modeling versatility, performance, and the training stability and efficiency of scaling Transformers.

- Stability - [**DeepNet**](https://arxiv.org/abs/2203.00555): Scaling Transformers to 1,000 Layers and beyond
- Versatility - [**Foundation Transformers (Magneto)**](https://arxiv.org/abs/2210.06423): Towards true all-purpose modeling across tasks and modalities (including language, vision, speech, and multimodal)
- Performance - A [**Length-Extrapolatable**](https://arxiv.org/abs/2212.10554) Transformer
- Efficiency - [**X-MoE**](https://arxiv.org/abs/2204.09179): Scalable & finetunable sparse Mixture-of-Experts (MoE)

## Latest

- November, 2022: Zeta 0.1.1 released [[Paper](https://arxiv.org/abs/2211.13184)] [[PyPI](https://pypi.org/project/zeta/)]

## Installation

To install:
```
pip install zeta
```

For local development:
```
git clone https://github.com/kygomez/zeta.git
cd zeta
pip install -e .
```

## Getting Started

With just a few lines of code, you can create a model with the above pioneering research features enabled. Here is a quick way to generate a BERT-like encoder:

```python
>>> from zeta.architecture.config import EncoderConfig
>>> from zeta.architecture.encoder import Encoder

>>> config = EncoderConfig(vocab_size=64000)
>>> model = Encoder(config)

>>> print(model)
```

We also support the `Decoder` architecture and the `EncoderDecoder` architecture:

```python
# Creating a decoder model
>>> from zeta.architecture.config import DecoderConfig
>>> from zeta.architecture.decoder import Decoder

>>> config = DecoderConfig(vocab_size=64000)
>>> decoder = Decoder(config)
>>> print(decoder)

# Creating a encoder-decoder model
>>> from zeta.architecture.config import EncoderDecoderConfig
>>> from zeta.architecture.encoder_decoder import EncoderDecoder

>>> config = EncoderDecoderConfig(vocab_size=64000)
>>> encdec = EncoderDecoder(config)
>>> print(encdec)
```

## Key Features

- [DeepNorm to enhance the training stability of Post-LayerNorm Transformers](https://arxiv.org/abs/2203.00555)
  * Enabled by setting *deepnorm=True* in the `Config` class. 
  * Adapts both the residual connection and the initialization method according to the model architecture (i.e., encoder, decoder, or encoder-decoder).

- [SubLN for model versatility and training stability](https://arxiv.org/abs/2210.06423)
  * Enabled by *subln=True*. This is activated by default. 
  * Introduces another LayerNorm to each sublayer and adjusts the initialization according to the model architecture.
  * Note that SubLN and DeepNorm cannot be used in one single model.

- [X-MoE: Efficient and finetunable sparse MoE modeling](https://arxiv.org/abs/2204.09179)
  * Enabled by *use_xmoe=True*. 
  * It replaces every *'moe_freq'* `FeedForwardNetwork` layers with the X-MoE layers.

- [Multiway architecture for multimodality](https://arxiv.org/abs/2208.10442)
  * Enabled by *multiway=True*.
  * It provides a pool of Transformer's parameters used for different modalities.

- [Extrapolatable position embedding (Xpos)](https://arxiv.org/abs/2212.10554)
  * Enabled by *xpos_rel_pos=True*.

- [Relative position bias](https://arxiv.org/abs/1910.10683)
  * Enabled by adjusting *rel_pos_buckets* and *max_rel_pos*.

- [SparseClip: Enhancing the gradient clipping for sparse MoE models](https://arxiv.org/abs/2211.13184)
  * We provide a [sample code](examples/fairseq/utils/sparse_clip.py) that can be easily adapted to the FairSeq (or other) repo.

Most of the features above can be activated by simply passing the corresponding parameters to the config. For example:

```python
>>> from zeta.architecture.config import EncoderConfig
>>> from zeta.architecture.encoder import Encoder

>>> config = EncoderConfig(vocab_size=64000, deepnorm=True, multiway=True)
>>> model = Encoder(config)

>>> print(model)
```

## Examples

We provide examples of how to utilize Zeta in the following scenarios/tasks:

- Language

  * [Decoder/GPT](examples/fairseq/README.md#example-gpt-pretraining)

  * [Encoder-Decoder/Neural Machine Translation](examples/fairseq/README.md#example-machine-translation)

  * [Encoder/BERT](examples/fairseq/README.md#example-bert-pretraining)

- Vision

  * ViT/BEiT [In progress]

- Speech

- Multimodal

  * [Multiway Transformers/BEiT-3](https://github.com/kygomez/unilm/tree/master/beit3)

We aim to add more examples regarding different tasks (e.g. vision pretraining and speech recognition) and various deep learning toolkits (e.g. [DeepSpeed](https://github.com/kygomez/DeepSpeed) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)). All feedback or PRs are welcome!

## Results

### Stability Evaluation

<p align="center">
  <img src="https://publicmodel.blob.core.windows.net/zeta/pic/convergence.png" width="800"/>
</p>

With Zeta, the training curve is smooth, while the baseline Transformer cannot converge.

### Scaling-up Experiments

<p align="center">
  <img src="https://publicmodel.blob.core.windows.net/zeta/pic/scaling_curve.png" width="800"/>
</p>

Zeta supports arbitrary depths and widths, scaling-up the models smoothly and efficiently.

## Acknowledgments

Certain implementations in Zeta have been adapted from or inspired by the [FairSeq](https://github.com/facebookresearch/fairseq) repository and the [UniLM](https://github.com/kygomez/unilm) repository.

## Citations

If you find this repository helpful, kindly consider citing our work:

```
@article{zeta,
  author    = {Shuming Ma and Hongyu Wang and Shaohan Huang and Wenhui Wang and Zew

# Zeta Halo - A Library for AI Transformers Across the Galactic Scale

<p>
  <a href="https://github.com/Kygomez/zeta-halo/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://pypi.org/project/zeta-halo"><img alt="Zeta Halo on PyPI" src="https://badge.fury.io/py/zeta-halo.svg" /></a>
</p>

Zeta Halo is a PyTorch library that empowers researchers and developers to ascend Transformers efficiently and effectively across the expanses of AI research.
It embodies crucial research implementations to enhance modeling versatility, capability, training stability, and efficiency of scaling Transformers.

- Stability - [**DeepNet**](https://arxiv.org/abs/2203.00555): Scaling Transformers to 1,000 Layers and beyond
- Versatility - [**Foundation Transformers (Magneto)**](https://arxiv.org/abs/2210.06423): Paving the way for true multipurpose modeling across various tasks and modalities (including language, vision, speech, and multimodal)
- Capability - A [**Length-Extrapolatable**](https://arxiv.org/abs/2212.10554) Transformer
- Efficiency - [**X-MoE**](https://arxiv.org/abs/2204.09179): Scalable & adjustable sparse Mixture-of-Experts (MoE)

## AI Event Timeline

- November, 2022: Zeta Halo 0.1.1 launched [[Paper](https://arxiv.org/abs/2211.13184)] [[PyPI](https://pypi.org/project/zeta-halo/)]

## Installation

To install, execute:

```
pip install zeta-halo
```

Or, to develop locally:

```
git clone https://github.com/kygomez/zeta-halo.git
cd zeta-halo
pip install -e .
```

## Startup Guide

It only takes a few lines of code to instantiate a model with the fundamental research features enabled. Here's how to quickly generate a BERT-like encoder:

```python
>>> from zeta_halo.architecture.config import EncoderConfig
>>> from zeta_halo.architecture.encoder import Encoder

>>> config = EncoderConfig(vocab_size=64000)
>>> model = Encoder(config)

>>> print(model)
```

We also support the `Decoder` architecture and the `EncoderDecoder` architecture:

```python
# Creating a decoder model
>>> from zeta_halo.architecture.config import DecoderConfig
>>> from zeta_halo.architecture.decoder import Decoder

>>> config = DecoderConfig(vocab_size=64000)
>>> decoder = Decoder(config)
>>> print(decoder)

# Creating a encoder-decoder model
>>> from zeta_halo.architecture.config import EncoderDecoderConfig
>>> from zeta_halo.architecture.encoder_decoder import EncoderDecoder

>>> config = EncoderDecoderConfig(vocab_size=64000)
>>> encdec = EncoderDecoder(config)
>>> print(encdec)
```

## Key Features

A comprehensive list of our features can be found in the original document. All features can be activated by simply passing the relevant parameters to the config. For instance:

```python
>>> from zeta_halo.architecture.config import EncoderConfig
>>> from zeta_halo.architecture.encoder import Encoder

>>> config = EncoderConfig(vocab_size=64000, deepnorm=True, multiway=True)
>>> model = Encoder(config)

>>> print(model)
```

## Examples

We provide examples on how to utilize Zeta Halo in different scenarios/tasks:

- Language
  * [Decoder/GPT](examples/fairseq/README.md#example-gpt-pretraining)
  * [Encoder-Decoder/Neural Machine Translation](examples/fairseq/README.md#example-machine-translation)
  * [Encoder/BERT](examples/fairseq/README.md#example-bert-pretraining)
- Vision
  * ViT/BEiT [In progress]
- Speech
- Multimodal
  * [Multiway Transformers/BEiT-3](https://github.com/Kygomez/unilm/tree/master/beit3)

More examples are forthcoming, focusing on different tasks (e.g. vision pretraining and speech recognition) and diverse deep learning toolkits. We welcome your comments and PRs!

## Results

### Stability Evaluation

<p align="center">
  <img src="https://publicmodel.blob.core.windows.net/zeta-halo/pic/convergence.png" width="800"/>
</p>

Zeta Halo ensures a smooth training curve, unlike the baseline Transformer that struggles to converge.

### Scaling-up Experiments

<p align="center">
  <img src="https://publicmodel.blob.core.windows.net/zeta-halo/pic/scaling_curve.png" width="800"/>
</p>

Zeta Halo supports limitless depths and widths, successfully scaling-up the models without hassle.

## Acknowledgments

Some implementations in Zeta Halo have either been adapted from or inspired by the [FairSeq](https://github.com/facebookresearch/fairseq) repository and the [UniLM](https://github.com/Kygomez/unilm) repository.

## Citations

Should you find this repository beneficial, please consider citing our work. Refer to the original document for the appropriate citation format for each component.

## Contributing

This project welcomes contributions and suggestions. Most contributions will require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and indeed do, grant us the rights to use your contribution. For details, visit https://cla.opensource.Kygomez.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Just follow the bot's instructions. You only need to do this once across all repos using our CLA.

This project follows the [Kygomez Open Source Code of Conduct](https://opensource.Kygomez.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.Kygomez.com/codeofconduct/faq/) or contact [Furu Wei](mailto:fuwei@Kygomez.com) and [Shuming Ma](mailto:shumma@Kygomez.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Kygomez trademarks or logos is subject to and must follow [Kygomez's Trademark & Brand Guidelines](https://www.Kygomez.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Kygomez trademarks or logos in modified versions of this project must not cause confusion or imply Kygomez sponsorship.
Any use of third-party trademarks or logos is subject to those third-party's policies.