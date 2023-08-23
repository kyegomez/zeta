[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Zeta - Seamlessly Create Zetascale Transformers


[![Docs](https://readthedocs.org/projects/zeta/badge/)](https://zeta.readthedocs.io)

<p>
  <a href="https://github.com/kyegomez/zeta/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://pypi.org/project/zetascale"><img alt="MIT License" src="https://badge.fury.io/py/zetascale.svg" /></a>
</p>

The Zeta framework provides developers with the ability to create State of The Art Multi-Modality Models Models as seamlessly as possible through an radical emphasis on Modularity, Reliability, Use-Ability, and Speed.

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

Creating a model empowered with the aforementioned breakthrough research features is a breeze. Here's how to quickly materialize the renowned Flash Attention

```python
import torch
from zeta import FlashAttention

q = torch.randn(2, 4, 6, 8)
k = torch.randn(2, 4, 10, 8)
v = torch.randn(2, 4, 10, 8)

attention = FlashAttention(causal=False, dropout=0.1, flash=True)
output = attention(q, k, v)

print(output.shape) 

```


## Acknowledgments

Zeta is a masterpiece inspired by LucidRains's repositories and elements of [FairSeq](https://github.com/facebookresearch/fairseq) and [UniLM](https://github.com/kyegomez/unilm).

## Citations

If our work here in Zeta has aided you in your journey, please consider acknowledging our efforts in your work. You can find relevant citation details in our [Citations Document](citations.md).

## Contributing
We're dependent on you for contributions, it's only Kye maintaining this repository and it's very difficult and with that said any contribution is infinitely appreciated by not just me but by Zeta's users who dependen on this repository to build the world's
best AI models

* Head over to the project board to look at open features to implement or bugs to tackle


## Todo
* Head over to the project board to look at open features to implement or bugs to tackle
