# top_p

# Module Name: zeta.utils.top_p

Function: 
```python
def top_p(logits, thres=0.9):
```

The `top_p` function is a part of the `zeta.utils` library. This function uses a process known as nucleus sampling, or top-p sampling, to handle logits from a language model. This function is intended to be used with the softmax output of language model sequences, making it an important method for text generation tasks.

Nucleus sampling is a form of sampling to solve the problem of text generation. It selects the highest probability tokens whose cumulative probability mass exceeds a given threshold.

This function is especially useful for deep learning algorithms involved in text generation tasks, where using pure maximum likelihood approximations might lead to highly repetitive and nonsensical outputs. By applying the `top_p` function, we can ensure more diverse and sensible outputs from such text generation models.

## Parameters:

Name | Type | Description | Default Value
--- | --- | --- | ---
logits | Tensor | These are the model's output log probabilities, expected to be in the format of a 2D tensor. ||
thres | float | A hyperparameter for top-p sampling, it adjusts the trade-off between randomness and fidelity in the generated text. This parameter indicates the cumulative probability threshold used for the nucleus sampling. | 0.9

The function returns logits processed by top-p sampling method, with least probable options removed according to the defined threshold value.

## Usage 

For this function, we first begin by importing the necessary libraries, which in this case are `torch` and its sublibrary `torch.nn.functional`.

``` python
import torch
import torch.nn.functional as F

def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)
```

We can illustrate the process using a simple example.

``` python
# Define logits tensor         
logits = torch.tensor([[0.5, 0.4, 0.1]]) 

# Call the top_p function   
filtered_logits = top_p(logits, thres=0.9)
print('The filtered logits are:')
print(filtered_logits)

# this should give us:
# tensor([[[0.5000], [0.4000], [-inf.]])
```

In this example, `'filtered_logits'` now contains the logits from `'logits'` but the least probable entries (inferior to `thres`) have been replaced by `-inf.` which makes them impossible to be chosen in a subsequent random sampling.

Keep in mind that in actual use cases the logits tensor would be the output of a pretrained language model and would have more complex dimensions, but the function would be used in the same way.

## Tips
- The choice of threshold value `'thres'` in the function `top_p(logits, thres=0.9)` is very important, as it determines the trade-off between fidelity (how closely the generated text matches the given input text) and diversity (how different the generated text is from the input text). A smaller threshold value may lead to more repetitive and less diverse text, while a larger threshold value may lead to more diverse but also more unpredictable and potentially incoherent text. You can fine-tune this value based on your specific needs and objectives.

## References
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)

Reference to PyTorch which this function is heavily tied to:

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) for further exploration.
