import torch
from zeta.models.gpt4 import GPT4

x = torch.randint(0, 256, (1, 1024)).cuda()

gpt4 = GPT4()

gpt4(x)
