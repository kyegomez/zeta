import torch
from zeta.tokenizers.tokenmonster import TokenMonster

tokenizer = TokenMonster("englishcode-32000-consistent-v1")

result = tokenizer.tokenize("Hello world!")

tensors = torch.tensor(result)
