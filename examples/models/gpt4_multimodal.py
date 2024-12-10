import torch

from zeta.models import GPT4MultiModal

image = torch.randint(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 1024))

model = GPT4MultiModal()
output = model(text, image)
