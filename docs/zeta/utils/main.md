# zeta.utils.main
Here are the helper functions and utils function used again and again in model engineering, all of these functions or classes can be imports from:

`from zeta.utils import x`


----

## Function: exists(val)
Check if the value is not None.

### Parameters:
- `val`: The value to check.

### Returns:
- `bool`: True if value exists (is not None), False otherwise.

### Example:
```python
from zeta.utils.main import exists

value1 = 10
value2 = None

print(exists(value1))  # Output: True
print(exists(value2))  # Output: False
```

## Function: default(val, d)
Return the value if it exists, otherwise return a default value.

### Parameters:
- `val`: The value to check.
- `d`: The default value to return if val is None.

### Returns:
- The value if it exists, otherwise the default value.

### Example:
```python
from zeta.utils.main import default

value1 = 5
value2 = None

result1 = default(value1, 0)  # Output: 5
result2 = default(value2, 0)  # Output: 0

print(result1)
print(result2)
```

## Function: once(fn)
Decorator to ensure the function is only called once.

### Parameters:
- `fn` (function): The function to wrap.

### Returns:
- `function`: The wrapped function.

### Example:
```python
from zeta.utils.main import once

@once
def perform_operation():
    print("Operation performed")

perform_operation()  # Output: Operation performed
perform_operation()  # No output (function is only called once)
```

## Function: eval_decorator(fn)
Decorator to ensure a method switches to eval mode before execution and returns to its original mode afterwards. For torch.nn.Module objects.

### Parameters:
- `fn` (function): The function to wrap.

### Returns:
- `function`: The wrapped function.

### Example:
```python
from zeta.utils.main import eval_decorator
import torch
import torch.nn as nn

class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    @eval_decorator
    def forward(self, x):
        return x

model = ExampleModel()
model.train()  # Set model to training mode
output = model(torch.tensor([1, 2, 3]))
print(output)  # Output: tensor([1, 2, 3])
model.eval()  # Set model to evaluation mode
output = model(torch.tensor([4, 5, 6]))
print(output)  # Output: tensor([4, 5, 6])
```

## Function: cast_tuple(val, depth)
Cast a value to a tuple of a specific depth.

### Parameters:
- `val`: Value to be cast.
- `depth` (int): Depth of the tuple.

### Returns:
- `tuple`: Tuple of the given depth with repeated val.

### Example:
```python
from zeta.utils.main import cast_tuple

value = 5
depth = 3

result = cast_tuple(value, depth)  # Output: (5, 5, 5)
print(result)
```

## Function: maybe(fn)
Decorator that calls a function if the first argument exists.

### Parameters:
- `fn` (function): The function to wrap.

### Returns:
- `function`: The wrapped function.

### Example:
```python
from zeta.utils.main import maybe

@maybe
def perform_operation(x):
    print(f"Operation performed with {x}")

perform_operation(10)  # Output: Operation performed with 10
perform_operation(None)  # No output (function not called)
```

## Class: always
Class that always returns a specified value when called.

### Parameters:
- `val`: The value to always return.

### Methods:
- `__call__(*args, **kwargs)`: Return the specified value.

### Example:
```python
from zeta.utils.main import always

always_5 = always(5)
result = always_5()  # Output: 5
print(result)
```

## Class: not_equals
Class that checks if a value does not equal the specified value.

### Parameters:
- `val`: The value to compare against.

### Methods:
- `__call__(x, *args, **kwargs)`: Compare the input x with the specified value.

### Example:
```python
from zeta.utils.main import not_equals

not_five = not_equals(5)
result1 = not_five(5)  # Output: False
result2 = not_five(10)  # Output: True

print(result1)
print(result2)
```

## Class: equals
Class that checks if a value equals the specified value.

### Parameters:
- `val`: The value to compare against.

### Methods:
- `__call__(x, *args, **kwargs)`: Compare the input x with the specified value.

### Example:
```python
from zeta.utils.main import equals

is_five = equals(5)
result1 = is_five(5)  # Output: True
result2 = is_five(10)  # Output: False

print(result1)
print(result2)
```

## Function: init_zero_(layer)
Initialize the weights and bias of a torch layer to zero.

### Parameters:
- `layer` (torch.nn.Module): The layer to initialize.

### Example:
```python
from zeta.utils.main import init_zero_
import torch.nn as nn

layer = nn.Linear(10, 5)
init_zero_(layer)

print(layer.weight)
print(layer.bias)
```

## Function: pick_and_pop(keys, d)
Remove and return values from a dictionary based on provided keys.

### Parameters:
- `keys` (list): List of keys to remove from the dictionary.
- `d` (dict): The dictionary to pick from.

### Returns:
- `dict`: A dictionary with the specified keys and their values.

### Example:
```python
from zeta.utils.main

 import pick_and_pop

data = {'a': 1, 'b': 2, 'c': 3}
keys = ['a', 'c']

result = pick_and_pop(keys, data)  # Output: {'a': 1, 'c': 3}
print(result)
print(data)  # Output: {'b': 2} (keys 'a' and 'c' removed)
```

## Function: group_dict_by_key(cond, d)
Group dictionary keys based on a condition.

### Parameters:
- `cond` (function): Condition to split dictionary.
- `d` (dict): The dictionary to group.

### Returns:
- `tuple`: Two dictionaries split based on the condition.

### Example:
```python
from zeta.utils.main import group_dict_by_key

data = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
condition = lambda x: x in ['a', 'b']

group1, group2 = group_dict_by_key(condition, data)
print(group1)  # Output: {'a': 1, 'b': 2}
print(group2)  # Output: {'c': 3, 'd': 4}
```

## Function: string_begins_with(prefix, str)
Check if a string begins with a specific prefix.

### Parameters:
- `prefix` (str): The prefix to check for.
- `str` (str): The string to check.

### Returns:
- `bool`: True if string starts with prefix, False otherwise.

### Example:
```python
from zeta.utils.main import string_begins_with

result1 = string_begins_with('hello', 'hello world')  # Output: True
result2 = string_begins_with('world', 'hello world')  # Output: False

print(result1)
print(result2)
```

## Function: group_by_key_prefix(prefix, d)
Group dictionary items by keys that start with a specific prefix.

### Parameters:
- `prefix` (str): The prefix to check for.
- `d` (dict): The dictionary to group.

### Returns:
- `tuple`: Two dictionaries split based on the prefix condition.

### Example:
```python
from zeta.utils.main import group_by_key_prefix

data = {'prefix_a_1': 1, 'prefix_a_2': 2, 'prefix_b_1': 3}
prefix = 'prefix_a'

group1, group2 = group_by_key_prefix(prefix, data)
print(group1)  # Output: {'prefix_a_1': 1, 'prefix_a_2': 2}
print(group2)  # Output: {'prefix_b_1': 3}
```

## Function: groupby_prefix_and_trim(prefix, d)
Group dictionary items by keys that start with a specific prefix and remove the prefix.

### Parameters:
- `prefix` (str): The prefix to check for.
- `d` (dict): The dictionary to group.

### Returns:
- `tuple`: Dictionary with the prefix removed and another dictionary with remaining items.

### Example:
```python
from zeta.utils.main import groupby_prefix_and_trim

data = {'prefix_a_1': 1, 'prefix_a_2': 2, 'prefix_b_1': 3}
prefix = 'prefix_a'

group1, group2 = groupby_prefix_and_trim(prefix, data)
print(group1)  # Output: {'1': 1, '2': 2}
print(group2)  # Output: {'prefix_b_1': 3}
```

## Function: divisible_by(num, den)
Check if a number is divisible by another number.

### Parameters:
- `num` (int): The number to check for divisibility.
- `den` (int): The divisor.

### Returns:
- `bool`: True if num is divisible by den, False otherwise.

### Example:
```python
from zeta.utils.main import divisible_by

result1 = divisible_by(10, 2)  # Output: True
result2 = divisible_by(7, 3)   # Output: False

print(result1)
print(result2)
```

## Function: top_p(logits, thres = 0.9)
Apply top-p sampling to logits.

### Parameters:
- `logits` (torch.Tensor): Input logits.
- `thres` (float): Threshold value for top-p sampling.

### Returns:
- `torch.Tensor`: Processed logits.

### Example:
```python
from zeta.utils.main import top_p
import torch

logits = torch.tensor([1.0, 2.0, 3.0])
processed_logits = top_p(logits)  # Processed logits based on top-p sampling

print(processed_logits)
```

## Function: top_k(logits, thres=0.9)
Apply top-k sampling to logits.

### Parameters:
- `logits` (torch.Tensor): Input logits.
- `thres` (float): Threshold value for top-k sampling.

### Returns:
- `torch.Tensor`: Processed logits.

### Example:
```python
from zeta.utils.main import top_k
import torch

logits = torch.tensor([1.0, 2.0, 3.0])
processed_logits = top_k(logits)  # Processed logits based on top-k sampling

print(processed_logits)
```

## Function: top_a(logits, min_p_pow=2.0, min_p_ratio=0.02)
Apply top-a sampling to logits.

### Parameters:
- `logits` (torch.Tensor): Input logits.
- `min_p_pow` (float): Minimum probability power.
- `min_p_ratio` (float): Minimum probability ratio.

### Returns:
- `torch.Tensor`: Processed logits.

### Example:
```python
from zeta.utils.main import top_a
import torch

logits = torch.tensor([1.0, 2.0, 3.0])
processed_logits = top_a(logits)  # Processed logits based on top-a sampling

print(processed_logits)
```

## Function: log(t, eps=1e-20)
Compute the natural logarithm of a tensor element-wise.

### Parameters:
- `t` (torch.Tensor): Input tensor.
- `eps` (float): Epsilon value to prevent taking the log of zero.

### Returns:
- `torch.Tensor`: Natural logarithm of the input tensor.

### Example:
```python
from zeta.utils.main import log
import torch

tensor = torch.tensor([0.5, 1.0, 2.0])
log_tensor = log(tensor)  # Output: tensor([-0.6931,  0.0000,  0.6931])

print(log_tensor)
```

## Function: gumbel_noise(t)
Generate Gumbel noise from a uniform noise tensor.

### Parameters:
- `t` (torch.Tensor): Input uniform noise tensor.

### Returns:
- `torch.Tensor`: Gumbel noise tensor.

### Example:
```python
from zeta.utils.main import gumbel_noise
import torch

uniform_noise = torch.rand(3)
gumbel_noise_tensor = gumbel_noise(uniform_noise)

print(gumbel_noise_tensor)
```

## Function: gumnel_sample(t, temperature=1., dim=-1)
Sample from a tensor using Gumbel-softmax relaxation.

### Parameters:
- `t` (torch.Tensor): Input tensor.
- `temperature` (float): Temperature parameter for sampling.
- `dim` (int): Dimension along which to apply Gumbel-softmax.

### Returns:
- `torch.Tensor`: Sampled tensor.

### Example:
```python
from zeta.utils.main import gumnel_sample
import torch

logits = torch.tensor([1.0, 2.0, 3.0])
sampled_tensor = gumnel_sample(logits)  # Sampled tensor using Gumbel-softmax

print(sampled_tensor)
```

## Class: ContrastiveTopK(nn.Module)
Calculate contrastive loss using top-k sampling.

### Parameters:
- `alpha`: Alpha value for contrastive loss.
- `k`: Number of top-k samples to consider.

### Methods:
- `forward(logits_exp, logits_ama)`: Calculate contrastive loss based on input logits.

### Example:
```python
from zeta.utils.main import ContrastiveTopK
import torch

contrastive = ContrastiveTopK(alpha=0.5, k=3)

logits_exp = torch.tensor([1.0, 2.0, 3.0])
logits_ama = torch.tensor([4.0, 5.0, 6.0])

loss = contrastive(logits_exp, logits_ama)
print(loss)
```

## Function: print_num_params(model, accelerator: Accelerator)
Print the number of parameters in a model.

### Parameters:
- `model`: The model to print parameter count for.
- `accelerator`: The accelerator object.

### Example:
```python
from zeta.utils.main import print_num_params
from accelerate import Accelerator
import torch.nn as nn

class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

model = ExampleModel()
accelerator = Accelerator()
print_num_params(model, accelerator)
```

## Class: Block(nn.Module)
A basic block module with convolution, normalization, and activation layers.

### Parameters:
- `dim` (int): Input dimension of the block.
- `dim_out` (int): Output dimension of the block.
- `groups` (int, optional): Number of groups for group normalization. Default is 8.

### Methods:
- `forward(x, scale_shift=None)`: Forward pass through the block.

### Example:
```python
from zeta.utils.main import Block
import torch

block = Block(dim=64, dim_out=128, groups=4)

x = torch.randn(1, 64, 16, 16)
output = block(x)

print(output.shape)
```

## Class: ResnetBlock(nn.Module)
A residual block with convolutional layers and optional time embedding.

### Parameters:
- `dim` (int): Input dimension of the block.
- `dim_out` (int): Output dimension of the block.
- `time_emb_dim` (int, optional): Dimension of the time embedding. Default is None.
- `groups` (int, optional): Number of groups for group normalization. Default is 8.

### Methods:
- `forward(x, time_emb=None)`: Forward pass through the block.

### Example:
```python
from zeta.utils.main import ResnetBlock
import torch

resnet_block = ResnetBlock(dim=128, dim_out=256, time_emb_dim=32)

x = torch.randn(1, 128, 8, 8)
time_emb = torch.randn(1, 32)
output = resnet_block(x, time_emb=time_emb)

print(output.shape)
```

## Function: load_model(path)
Load a model from a file.

### Parameters:
- `path` (str): Path to the file containing the model.

### Returns:
- `torch.nn.Module`: Loaded model.

### Example:
```python
from zeta.utils.main import load_model

model = load_model('model_checkpoint.pth')
print(model)
```

## Function: seek_all_images(img, channels=3)
Iterate over all frames of a GIF image.

### Parameters:
- `img` (PIL.Image.Image): Input GIF image.
- `channels` (int): Number of color channels. Default is 3.

### Yields:
- `PIL.Image.Image`: Frames of the GIF image.

### Example:
```python
from zeta.utils.main import seek_all_images
from PIL import Image

gif_path = 'animation.gif'
gif_img = Image.open(gif_path)

for frame in seek_all_images(gif_img, channels=3):
    frame.show()
```

## Function: video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True)
Convert a video tensor to a GIF image.

### Parameters:
- `tensor` (torch.Tensor): Video tensor of shape (channels, frames, height, width).
- `path` (str): Path to save the GIF image.
- `duration` (int): Duration of each frame in milliseconds. Default is 120.
- `loop` (int): Number of loops for the GIF. Default is 0 (infinite).
- `optimize` (bool): Whether to optimize the GIF for size. Default is True.

### Example:
```python
from zeta.utils.main import video_tensor_to_gif
import torch

video_tensor = torch.randn(3, 10, 256, 256)
output_gif_path = 'output_animation.gif'

video_tensor_to_gif(video_tensor, output_gif_path, duration=100)
```

## Function: gif_to_tensor(path, channels=3, transform=T.ToTensor())
Convert a GIF image to a video tensor.

### Parameters:
- `path` (str): Path to the GIF image.
- `channels` (int): Number of color channels. Default is 3.
- `transform` (callable): Transformation function to apply to each frame. Default is `T.ToTensor()`.

### Returns:
- `torch.Tensor`: Video tensor of shape (channels, frames, height, width).

### Example:
```python
from zeta.utils.main import gif_to_tensor

input_gif_path = 'input_animation.gif'
video_tensor = gif_to_tensor(input_gif_path, channels=3)

print(video_tensor.shape)
```

## Function: identity(t, *args, **kwargs)
Identity function that returns the input tensor as is.

### Parameters:
- `t` (torch.Tensor): Input tensor.
- `*args` (tuple): Additional positional arguments.
- `**kwargs` (dict): Additional keyword arguments.

### Returns:
- `torch.Tensor`: Input tensor.

### Example:
```python
from zeta.utils.main import identity
import torch

tensor = torch.tensor([1.0, 2.0, 3.0])
output = identity(tensor, some_arg='value')

print(output)
```

## Function: normalize_img(t)
Normalize an image tensor to the range [-1, 1].

### Parameters:
- `t` (torch.Tensor): Input image tensor.

### Returns:
- `torch.Tensor`: Normalized image tensor.

### Example:
```python
from zeta.utils.main import normalize_img
import torch

image_tensor = torch.rand(3, 256, 256)  # RGB image
normalized_image = normalize_img(image_tensor)

print(normalized_image.min(), normalized_image.max())
```

## Function: unnormalize_img(t)
Unnormalize a normalized image tensor.

### Parameters:
- `t` (torch.Tensor): Input normalized image tensor.

### Returns:
- `torch.Tensor`: Unnormalized image tensor.

### Example:
```python
from zeta.utils.main import unnormalize_img
import torch

normalized_image = torch.rand(3, 256, 256)  # Normalized image
unnormalized_image = unnormalize_img(normalized_image)

print(unnormalized_image.min(), unnormalized_image.max())
```

## Function: cast_num_frames(t, frames)
Cast the number of frames in a video tensor to a specific value.

### Parameters:
- `t` (torch.Tensor): Input video tensor of shape (channels, frames, height, width).
- `frames` (int): Number of frames to cast to.

### Returns:
- `torch.Tensor`: Video tensor with the specified number of frames.

### Example:
```python
from zeta.utils.main import cast_num_frames
import torch

video_tensor = torch.rand(3, 10, 256, 256)
video_tensor_casted = cast_num_frames(video_tensor, frames=8)

print(video_tensor_casted.shape)
```

## Function: max_neg_values(tensor)
Get the maximum negative value for a tensor's data type.

### Parameters:
- `tensor` (torch.Tensor): Input tensor.

### Returns:
- `float`: Maximum negative value.

### Example:
```python
from zeta.utils.main import max_neg_values
import torch

tensor = torch.tensor([1.0, 2.0, 3.0])
max_neg = max_neg_values(tensor.dtype)

print(max_neg)
```

## Function: l2norm(t, groups=1)
Perform L2 normalization along specified groups of a tensor.

### Parameters:
- `t` (torch.Tensor): Input tensor.
- `groups` (int): Number of groups

 for normalization. Default is 1.

### Returns:
- `torch.Tensor`: L2 normalized tensor.

### Example:
```python
from zeta.utils.main import l2norm
import torch

tensor = torch.tensor([1.0, 2.0, 3.0])
l2_normalized_tensor = l2norm(tensor, groups=2)

print(l2_normalized_tensor)
```

## Function: pad_at_dim(t, pad, dim=-1, value=0.)
Pad a tensor along a specified dimension.

### Parameters:
- `t` (torch.Tensor): Input tensor.
- `pad` (tuple): Padding values to add before and after the dimension.
- `dim` (int): Dimension along which to pad. Default is -1.
- `value` (float): Padding value. Default is 0.

### Returns:
- `torch.Tensor`: Padded tensor.

### Example:
```python
from zeta.utils.main import pad_at_dim
import torch

tensor = torch.tensor([1.0, 2.0, 3.0])
padded_tensor = pad_at_dim(tensor, pad=(1, 1), dim=-1, value=-1)

print(padded_tensor)
```

## Function: or_reduce(masks)
Perform element-wise logical OR reduction on a list of masks.

### Parameters:
- `masks` (list of torch.Tensor): List of boolean masks.

### Returns:
- `torch.Tensor`: Resulting mask after OR reduction.

### Example:
```python
from zeta.utils.main import or_reduce
import torch

mask1 = torch.tensor([True, False, True])
mask2 = torch.tensor([False, True, False])
result_mask = or_reduce([mask1, mask2])

print(result_mask)
```

## Class: Residual(nn.Module)
A wrapper module that adds residual connections to a given module.

### Parameters:
- `fn` (nn.Module): Module to wrap with residual connection.

### Methods:
- `forward(x, *args, **kwargs)`: Forward pass through the module with residual connection.

### Example:
```python
from zeta.utils.main import Residual
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        # Define your layers here
    
    def forward(self, x):
        # Forward pass logic
        
my_module = MyModule()
residual_module = Residual(my_module)

x = torch.randn(1, 64)
output = residual_module(x)

print(output.shape)
```

## Class: SinusoidalPosEmb(nn.Module)
Sinusoidal positional embedding module for self-attention mechanisms.

### Parameters:
- `dim` (int): Dimension of the positional embedding.

### Methods:
- `forward(x)`: Forward pass to generate positional embeddings for input tensor.

### Example:
```python
from zeta.utils.main import SinusoidalPosEmb
import torch

pos_emb_module = SinusoidalPosEmb(dim=128)

x = torch.randn(1, 16, 128)  # Input tensor
pos_emb = pos_emb_module(x)

print(pos_emb.shape)
```

## Function: upsample(dim)
Create an upsample layer for a given dimension.

### Parameters:
- `dim` (int): Dimension of the input and output channels.

### Returns:
- `nn.Module`: Upsample layer.

### Example:
```python
from zeta.utils.main import upsample
import torch.nn as nn

upsample_layer = upsample(dim=256)

x = torch.randn(1, 256, 8, 8)  # Input tensor
output = upsample_layer(x)

print(output.shape)
```

## Function: downsample(dim)
Create a downsample layer for a given dimension.

### Parameters:
- `dim` (int): Dimension of the input and output channels.

### Returns:
- `nn.Module`: Downsample layer.

### Example:
```python
from zeta.utils.main import downsample
import torch.nn as nn

downsample_layer = downsample(dim=256)

x = torch.randn(1, 256, 16, 16)  # Input tensor
output = downsample_layer(x)

print(output.shape)
```

## Class: LayerNorm(nn.Module)
Layer normalization module.

### Parameters:
- `dim` (int): Dimension for normalization.
- `eps` (float): Small value added to the denominator for numerical stability.

### Methods:
- `forward(x)`: Forward pass through the layer normalization.

### Example:
```python
from zeta.utils.main import LayerNorm
import torch.nn as nn

layer_norm = LayerNorm(dim=256, eps=1e-5)

x = torch.randn(1, 256, 16, 16)  # Input tensor
normalized_x = layer_norm(x)

print(normalized_x.shape)
```

## Class: PreNorm(nn.Module)
Pre-normalization wrapper module.

### Parameters:
- `dim` (int): Dimension for normalization.
- `fn` (nn.Module): Module to wrap with pre-normalization.

### Methods:
- `forward(x, **kwargs)`: Forward pass through the module with pre-normalization.

### Example:
```python
from zeta.utils.main import PreNorm
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        # Define your layers here
    
    def forward(self, x):
        # Forward pass logic
        
my_module = MyModule()
pre_norm_module = PreNorm(dim=128, fn=my_module)

x = torch.randn(1, 128)
output = pre_norm_module(x)

print(output.shape)
```

## Function: cosine_beta_schedule(timesteps, s=0.008)
Generate a cosine beta schedule for progressive loss scaling.

### Parameters:
- `timesteps` (int): Total number of time steps.
- `s` (float): Scaling factor for the cosine function.

### Returns:
- `torch.Tensor`: Beta values for each time step.

### Example:
```python
from zeta.utils.main import cosine_beta_schedule
import torch

beta_schedule = cosine_beta_schedule(timesteps=1000, s=0.01)
print(beta_schedule)
```

## Class: Normalize(nn.Module)
Normalization module to perform L2 normalization along a specific dimension.

### Parameters:
- `dim` (int): Dimension for normalization.

### Methods:
- `forward(x)`: Forward pass through the normalization.

### Example:
```python
from zeta.utils.main import Normalize
import torch.nn as nn

normalize_module = Normalize(dim=256)

x = torch.randn(1, 256, 16, 16)  # Input tensor
normalized_x = normalize_module(x)

print(normalized_x.shape)
```

## Class: LearnableLogitScaling(nn.Module)
Learnable logit scaling module for temperature scaling in temperature sampling.

### Parameters:
- `logit_scale_init` (float): Initial value for the logit scale.
- `learnable` (bool): Whether the logit scale is learnable. Default is True.
- `max_logit_scale` (float): Maximum value for the logit scale. Default is 100.

### Methods:
- `forward(x)`: Forward pass through the learnable logit scaling.

### Example:
```python
from zeta.utils.main import LearnableLogitScaling
import torch.nn as nn

logit_scaling = LearnableLogitScaling(logit_scale_init=1.0, learnable=True, max_logit_scale=10.0)

x = torch.randn(1, 256)  # Input tensor
scaled_x = logit_scaling(x)

print(scaled_x.shape)


```

## Class: EinOpsRearrange(nn.Module)
EinOps-based module for rearranging tensor dimensions.

### Parameters:
- `rearrange_expr` (str): Rearrangement expression.
- `**kwargs`: Additional arguments for einops.rearrange.

### Methods:
- `forward(x)`: Forward pass to rearrange the input tensor.

### Example:
```python
from zeta.utils.main import EinOpsRearrange
import torch

rearrange_module = EinOpsRearrange(rearrange_expr='b h w c -> b c h w', h=16, w=16)

x = torch.randn(1, 16, 16, 256)  # Input tensor
rearranged_x = rearrange_module(x)

print(rearranged_x.shape)
```




------

## Function: get_sinusoid_encoding_table(n_position, d_hid)
Generate a sinusoidal positional encoding table for self-attention mechanisms.

### Parameters:
- `n_position` (int): Number of positions.
- `d_hid` (int): Hidden dimension.

### Returns:
- `torch.Tensor`: Sinusoidal positional encoding table.

### Example:
```python
from zeta.utils.main import get_sinusoid_encoding_table
import torch

pos_encoding_table = get_sinusoid_encoding_table(n_position=100, d_hid=128)

print(pos_encoding_table.shape)
```

## Function: interpolate_pos_encoding_2d(target_spatial_size, pos_embed)
Interpolate 2D positional embeddings to a target spatial size.

### Parameters:
- `target_spatial_size` (int): Target spatial size.
- `pos_embed` (torch.Tensor): Input positional embeddings.

### Returns:
- `torch.Tensor`: Interpolated positional embeddings.

### Example:
```python
from zeta.utils.main import interpolate_pos_encoding_2d
import torch

pos_embed = torch.randn(1, 64, 128)  # Input positional embeddings
interpolated_pos_embed = interpolate_pos_encoding_2d(target_spatial_size=256, pos_embed=pos_embed)

print(interpolated_pos_embed.shape)
```

## Function: cast_if_src_dtype(tensor, src_dtype, tgt_dtype)
Cast a tensor to a target dtype if its source dtype matches.

### Parameters:
- `tensor` (torch.Tensor): Input tensor.
- `src_dtype` (torch.dtype): Source dtype to check.
- `tgt_dtype` (torch.dtype): Target dtype to cast to.

### Returns:
- `torch.Tensor`: Casted tensor if necessary.

### Example:
```python
from zeta.utils.main import cast_if_src_dtype
import torch

tensor = torch.randn(1, 256)
casted_tensor = cast_if_src_dtype(tensor, src_dtype=torch.float32, tgt_dtype=torch.bfloat16)

print(casted_tensor.dtype)
```

## Class: SelectElements(nn.Module)
Select specific elements from an input tensor using given indices.

### Parameters:
- `index` (int): Index to select elements along a specific dimension.

### Methods:
- `forward(x)`: Forward pass to select elements from the input tensor.

### Example:
```python
from zeta.utils.main import SelectElements
import torch

select_module = SelectElements(index=2)

x = torch.randn(1, 4, 256)  # Input tensor
selected_elements = select_module(x)

print(selected_elements.shape)
```

## Class: SelectEOSAndProject(nn.Module)
Select elements from the end of a sequence and apply a projection.

### Parameters:
- `proj` (nn.Module): Projection module to apply after selection.

### Methods:
- `forward(x, seq_len)`: Forward pass to select elements and apply projection.

### Example:
```python
from zeta.utils.main import SelectEOSAndProject
import torch.nn as nn

proj_module = nn.Linear(256, 128)
select_and_project = SelectEOSAndProject(proj=proj_module)

x = torch.randn(1, 16, 256)  # Input tensor
seq_len = torch.tensor([10])  # Sequence length
output = select_and_project(x, seq_len)

print(output.shape)
```


