[![Multi-Modality](images/agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Zeta - Seamlessly Create Zetascale Transformers


[![Docs](https://readthedocs.org/projects/zeta/badge/)](https://zeta.readthedocs.io)

<p>
  <a href="https://github.com/kyegomez/zeta/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://pypi.org/project/zetascale"><img alt="MIT License" src="https://badge.fury.io/py/zetascale.svg" /></a>
</p>

Create Ultra-Powerful Multi-Modality Models Seamlessly and Efficiently in as minimal lines of code as possible.

# 🤝 Schedule a 1-on-1 Session
Book a [1-on-1 Session with Kye](https://calendly.com/apacai/agora), the Creator, to discuss any issues, provide feedback, or explore how we can improve Zeta for you.


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

# Documentation
[Click here for the documentation, it's at zeta.apac.ai](https://zeta.apac.ai)

# Vision
Zeta hopes to be the leading framework and library to effortlessly enable you to create the most capable and reliable foundation models out there with infinite scalability.

## Acknowledgments
Zeta is a masterpiece inspired by LucidRains's repositories and elements of [FairSeq](https://github.com/facebookresearch/fairseq) and [UniLM](https://github.com/kyegomez/unilm).


## Contributing
We're dependent on you for contributions, it's only Kye maintaining this repository and it's very difficult and with that said any contribution is infinitely appreciated by not just me but by Zeta's users who dependen on this repository to build the world's
best AI models

* Head over to the project board to look at open features to implement or bugs to tackle


## Todo
* Head over to the project board to look at open features to implement or bugs to tackle

## Project Board
[This weeks iteration is here](https://github.com/users/kyegomez/projects/7/views/2)
