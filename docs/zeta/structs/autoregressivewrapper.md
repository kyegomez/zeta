# AutoRegressiveWrapper Class

In the following documentation, you'll learn all about the AutoRegressiveWrapper class of zeta.structs module. As autoregressive models are sequence models used to predict subsequent data points in sequence data, this class provides a wrapper that can be used to wrap any PyTorch nn.Module to make them autoregressive model compliant.

## Table of Contents

1. Class Definition
2. Parameters
3. Methods
4. Examples
5. Conclusion

## 1. Class Definition

AutoRegressiveWrapper is a Python class that inherits from PyTorch's nn.Module and applies an autoregressive mask on the input sequence to any module that takes sequence input. This wrapper ensures the output sequence obeys a property inherent to causal or autoregressive models – the prediction at each position in the sequence is based only on preceding positions.

```python
class AutoRegressiveWrapper(nn.Module):
```

## 2. Parameters

The parameters accepted by AutoRegressiveWrapper are:

| Name | Type | Description | Default |
|---|---|---|---|
|net|nn.Module|A PyTorch module that takes a sequence of tokens and outputs a sequence of logits.|N/A|
|ignore_index|int|The index to ignore in the target sequence when calculating the loss.|-100|
|pad_value|int|The value to pad the target sequence with.|0|
|mask_prob|float|The probability of masking a token in the input sequence.|0.0|
|speculative |bool|Whether to use speculative decoding or not.|False|

## 3. Methods

The methods provided by AutoRegressiveWrapper are:

### 3.1 __init__()

The `__init__()` method initializes an instance of the AutoRegressiveWrapper class.

```python
def __init__(self, net, ignore_index=-100, pad_value=0, mask_prob=0.0, speculative=False)
```

### 3.2 forward()

The `forward()` method performs forward pass of the autoregressive wrapper.

```python
def forward(self, x, return_loss=True, **kwargs)
```

This method returns logits produced by the wrapped module. If `return_loss` is `True`, it also returns the loss calculated using target sequence and outputs of the wrapped module.

### 3.3 generate()

The `generate()` method generates a sequence of tokens from the model.

```python
def generate(self, start_tokens, seq_len, eos_token=None, strategy="temperature", temperature=1.0, filter_logits_fn=top_k, filter_thres=0.9, min_p_pow=2.0, min_p_ratio=0.02, gamma=5, **kwargs)
```

You can control the sequence generation with various parameters like `strategy`, `temperature`, `filter_logits_fn` etc.

### 3.4 generate_n_solutions()

The `generate_n_solutions()` method generates n solutions from the model.

```python
def generate_n_solutions(self, start_tokens, n, seqlen, **kwargs)
```
This method is particularly useful for generating multiple forecasted sequence paths.

### 3.5 evaluate_and_select_best_solution()

The `evaluate_and_select_best_solution()` method evaluates the solutions based on a reward model and returns the best one.

```python
def evaluate_and_select_best_solution(self, solutions, reward_model)
```


## 4. Examples

To help you better understand the usage of this class, here are some examples.

First example demonstrates how to instantiate the AutoRegressiveWrapper over an existing nn.module (nn.Linear in this case).

```python
import torch
import torch.nn as nn

from zeta.structs import AutoRegressiveWrapper

net = nn.Linear(10, 10)
net = AutoRegressiveWrapper(net)
x = torch.randn(1, 10)
logits, loss = net(x, return_loss=True)
print(logits.shape)
# Output: torch.Size([1, 10, 10]) # (batch_size, seq_len, vocab_size)
```

The second example demonstrates the usage of generate method to generate a sequence with the model.

```python
start_tokens = torch.tensor([1, 2, 3])
generated_sequence = net.generate(start_tokens, seq_len=10)
```
This generated_sequence represents the next 10 steps in the sequence (based on the first 3 steps provided as start_tokens).

The third example shows generating multiple solutions and selecting the best one.

```python
solutions = net.generate_n_solutions(start_tokens, n=5, seqlen=10)
best_solution = net.evaluate_and_select_best_solution(
    solutions, reward_model=lambda x: -x.sum()
)
```
In the example above, the reward model simply returns the negative sum of the sequence, and the solution with lowest sum is selected as the best solution.

## 5. Conclusion

In this documentation, you have learned about the AutoRegressiveWrapper class of zeta.structs. You should now be more comfortable and confident in leveraging this class in your neural network architectures to realize autoregressive transformation.
