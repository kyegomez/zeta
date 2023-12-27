# cosine_beta_schedule

# Module/Function Name: cosine_beta_schedule

Function `zeta.utils.cosine_beta_schedule(timesteps, s=0.008)` is a utility function in Zeta library that generates a cosine beta scheduler. This is done by creating an array where its values are incremented in a cosine manner between 0 and 1. Such schedule is often used in various applications such as learning rate scheduling in deep learning, simulating annealing schedule etc.

## Definition

```python
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = (
        torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)
```

## Parameters

| Parameters | Type | Description |
|-|-|-|
| timesteps | int | The total timesteps or epochs for the training or the annealing process |
| s | float, optional | The offset for the cosine function, default is `0.008` |

## Output

Returns a torch tensor of size `timesteps` containing beta values that forms a cosine schedule.

## Usage

Here are 3 examples of how to use the `cosine_beta_schedule` function:

### Example 1

In this example, we're generating a cosine beta schedule for 10 timesteps without an offset.

```python
import torch
from zeta.utils import cosine_beta_schedule

timesteps = 10
cosine_schedule = cosine_beta_schedule(timesteps)
print(cosine_schedule)
```

### Example 2

In this example, we're generating a cosine beta schedule for a specific timeframe with a custom offset.

```python
import torch
from zeta.utils import cosine_beta_schedule

timesteps = 1000
offset = 0.005
cosine_schedule = cosine_beta_schedule(timesteps, s=offset)
print(cosine_schedule)
```

### Example 3

In this example, we're using cosine beta schedule as a learning rate scheduler in a PyTorch training loop
