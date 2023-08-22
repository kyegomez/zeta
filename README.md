[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Zeta - A Library for Zetascale Transformations
[![Docs](https://readthedocs.org/projects/swarms/badge/)](https://swarms.readthedocs.io)

Docs for [Zeta](https://github.com/kyegomez/swarms).

<p>
  <a href="https://github.com/kyegomez/zeta/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://pypi.org/project/zeta"><img alt="MIT License" src="https://badge.fury.io/py/zeta.svg" /></a>
</p>

Zeta is a PyTorch-powered library, forged in the heart of the Halo array, that empowers researchers and developers to scale up Transformers efficiently and effectively. It leverages seminal research advancements to enhance the generality, capability, and stability of scaling Transformers while optimizing training efficiency.

## Installation

To install:
```
pip install zetascale
```

To get hands-on and develop it locally:
```
git clone https://github.com/kyegomez/zeta.git
cd zeta
pip install -e .
```

## Initiating Your Journey

Creating a model empowered with the aforementioned breakthrough research features is a breeze. Here's how to quickly materialize a BERT-like encoder:

```python
>>> from zeta import EncoderConfig
>>> from zeta import Encoder

>>> config = EncoderConfig(vocab_size=64000)
>>> model = Encoder(config)

>>> print(model)
```

Additionally, we support the `Decoder` and `EncoderDecoder` architectures:

```python
# To create a decoder model
>>> from zeta import DecoderConfig
>>> from zeta import Decoder

>>> config = DecoderConfig(vocab_size=64000)
>>> decoder = Decoder(config)
>>> print(decoder)

# To create an encoder-decoder model
>>> from zeta import EncoderDecoderConfig
>>> from zeta import EncoderDecoder

>>> config = EncoderDecoderConfig(vocab_size=64000)
>>> encdec = EncoderDecoder(config)
>>> print(encdec)
```

## Key Features

Most of the transformative features mentioned below can be enabled by simply setting the corresponding parameters in the `config`:

```python
>>> from zeta import EncoderConfig
>>> from zeta import Encoder

>>> config = EncoderConfig(vocab_size=64000, deepnorm=True, multiway=True)
>>> model = Encoder(config)

>>> print(model)
```

For a complete overview of our key features, refer to our [Feature Guide](features.md).

## Examples

Discover how to wield Zeta in a multitude of scenarios/tasks, including but not limited to:

- Language
  * [Decoder/GPT](examples/fairseq/README.md#example-gpt-pretraining)
  * [Encoder-Decoder/Neural Machine Translation](examples/fairseq/README.md#example-machine-translation)
  * [Encoder/BERT](examples/fairseq/README.md#example-bert-pretraining)

- Vision
  * ViT/BEiT [In progress]

- Speech

- Multimodal
  * [Multiway Transformers/BEiT-3](https://github.com/kyegomez/unilm/tree/master/beit3)

We are working tirelessly to expand the collection of examples spanning various tasks (e.g., vision pretraining, speech recognition) and various deep learning frameworks (e.g., [DeepSpeed](https://github.com/kyegomez/DeepSpeed), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)). Your comments, suggestions, or contributions are welcome!

## Results

Check out our [Results Page](results.md) to witness Zeta's exceptional performance in Stability Evaluations and Scaling-up Experiments.

## Acknowledgments

Zeta is a masterpiece inspired by elements of [FairSeq](https://github.com/facebookresearch/fairseq) and [UniLM](https://github.com/kyegomez/unilm).

## Citations

If our work here in Zeta has aided you in your journey, please consider acknowledging our efforts in your work. You can find relevant citation details in our [Citations Document](citations.md).

## Contributing

We're always thrilled to welcome new ideas and improvements from the community. Please check our [Contributor's Guide](contributing.md) for more details about contributing.


* Create an modular omni-universal Attention class with flash multihead attention or regular mh or dilated attention -> then integrate into Decoder/ DecoderConfig


