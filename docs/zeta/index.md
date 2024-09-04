# Zeta

Build SOTA AI Models 80% faster with modular, high-performance, and scalable building blocks!

[![Docs](https://readthedocs.org/projects/zeta/badge/)](https://zeta.readthedocs.io)

<p>
  <a href="https://github.com/kyegomez/zeta/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://pypi.org/project/zetascale"><img alt="MIT License" src="https://badge.fury.io/py/zetascale.svg" /></a>
</p>

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

[![GitHub issues](https://img.shields.io/github/issues/kyegomez/zeta)](https://github.com/kyegomez/zeta/issues) [![GitHub forks](https://img.shields.io/github/forks/kyegomez/zeta)](https://github.com/kyegomez/zeta/network) [![GitHub stars](https://img.shields.io/github/stars/kyegomez/zeta)](https://github.com/kyegomez/zeta/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/zeta)](https://github.com/kyegomez/zeta/blob/main/LICENSE)[![GitHub star chart](https://img.shields.io/github/stars/kyegomez/zeta?style=social)](https://star-history.com/#kyegomez/zeta)[![Dependency Status](https://img.shields.io/librariesio/github/kyegomez/zeta)](https://libraries.io/github/kyegomez/zeta) [![Downloads](https://static.pepy.tech/badge/zeta/month)](https://pepy.tech/project/zetascale)

[![Join the Agora discord](https://img.shields.io/discord/1110910277110743103?label=Discord&logo=discord&logoColor=white&style=plastic&color=d7b023)![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/zeta)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20AI%20project:%20&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fzeta) [![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fzeta) [![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fzeta&title=&summary=&source=)

[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fzeta&title=zeta%20-%20the%20future%20of%20AI) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fzeta&t=zeta%20-%20the%20future%20of%20AI) [![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fzeta&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=zeta%20-%20the%20future%20of%20AI) [![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=Check%20out%20zeta%20-%20the%20future%20of%20AI%20%23zeta%20%23AI%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2Fzeta)

After building out thousands of neural nets and facing the same annoying bottlenecks of chaotic codebases with no modularity and low performance modules, Zeta needed to be born to enable me and others to quickly prototype, train, and optimize the latest SOTA neural nets and deploy them into production. 

Zeta places a radical emphasis on useability, modularity, and performance. Zeta is now currently employed in 100s of models across my github and across others. 
Get started below and LMK if you want my help building any model, I'm here for you 😊 💜 

Here’s the refined module tree for the Zeta framework without the leftover commit messages:

### Zeta Framework Module Tree:

```bash
zeta/
├── experimental/      # Contains experimental features for testing future capabilities
├── models/            # Houses model architectures and neural network definitions
├── nn/                # Core neural network layers and utilities for building models
├── ops/               # Low-level operations and mathematical functions
├── optim/             # Optimization algorithms for training
├── rl/                # Reinforcement learning components and tools
├── structs/           # Data structures and utilities for managing model states
├── tokenizers/        # Tokenization modules for processing data (text, etc.)
├── training/          # High-level training loops and training utilities
├── utils/             # General-purpose utilities and helper functions
└── __init__.py        # Initializes the framework and handles global imports
```

### Zeta’s Abstraction:

The Zeta framework abstracts over **PyTorch** and **CUDA**, aiming to provide flexibility in building, training, and deploying models. Each module serves a distinct role, allowing users to construct neural networks, define custom operations, and handle everything from low-level ops to high-level training routines.

This structure provides several key benefits:
- **Modularity**: Each module encapsulates specific functionality, making it easy to extend or modify.
- **Flexibility**: Zeta integrates seamlessly with PyTorch and CUDA, but offers a more structured and organized way to build models.
- **Performance**: By building on CUDA, Zeta ensures efficient computation while maintaining ease of use through PyTorch's high-level abstractions.

This tree structure reflects the framework’s intent to simplify complex deep learning operations while providing the flexibility to customize each layer and operation according to specific use cases.