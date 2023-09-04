# Utils
All of the following funcs and classes can be imported from zeta like this

`from zeta.utils import x`

---

## exists(val)
Check if the value is not None.

**Parameters:**
- `val`: The value to check.

**Returns:**
- `bool`: True if value exists (is not None), False otherwise.

---

## default(val, d)
Return the value if it exists, otherwise return a default value.

**Parameters:**
- `val`: The value to check.
- `d`: The default value to return if val is None.

**Returns:**
- The value if it exists, otherwise the default value.

---

## once(fn)
Decorator to ensure the function is only called once.

**Parameters:**
- `fn` (function): The function to wrap.

**Returns:**
- `function`: The wrapped function.

---

## eval_decorator(fn)
Decorator to ensure a method switches to eval mode before execution and returns to its original mode afterwards. For torch.nn.Module objects.

**Parameters:**
- `fn` (function): The function to wrap.

**Returns:**
- `function`: The wrapped function.

---

## cast_tuple(val, depth)
Cast a value to a tuple of a specific depth.

**Parameters:**
- `val`: Value to be cast.
- `depth` (int): Depth of the tuple.

**Returns:**
- `tuple`: Tuple of the given depth with repeated val.

---

## maybe(fn)
Decorator that calls a function if the first argument exists.

**Parameters:**
- `fn` (function): The function to wrap.

**Returns:**
- `function`: The wrapped function.

---

## always()
Class that always returns a specified value when called.

**Parameters:**
- `val`: The value to always return.

**Returns:**
- The specified value.

---

## not_equals()
Class that checks if a value does not equal the specified value.

**Parameters:**
- `val`: The value to compare against.

**Returns:**
- `bool`: True if x is not equal to the specified value, False otherwise.

---

## equals()
Class that checks if a value equals the specified value.

**Parameters:**
- `val`: The value to compare against.

**Returns:**
- `bool`: True if x is equal to the specified value, False otherwise.

---

## init_zero_(layer)
Initialize the weights and bias of a torch layer to zero.

**Parameters:**
- `layer` (torch.nn.Module): The layer to initialize.

---

## pick_and_pop(keys, d)
Remove and return values from a dictionary based on provided keys.

**Parameters:**
- `keys` (list): List of keys to remove from the dictionary.
- `d` (dict): The dictionary to pick from.

**Returns:**
- `dict`: A dictionary with the specified keys and their values.

---

## group_dict_by_key(cond, d)
Group dictionary keys based on a condition.

**Parameters:**
- `cond` (function): Condition to split dictionary.
- `d` (dict): The dictionary to group.

**Returns:**
- `tuple`: Two dictionaries split based on the condition.

---

## string_begins_with(prefix, str)
Check if a string begins with a specific prefix.

**Parameters:**
- `prefix` (str): The prefix to check for.
- `str` (str): The string to check.

**Returns:**
- `bool`: True if string starts with prefix, False otherwise.

---

## group_by_key_prefix(prefix, d)
Group dictionary items by keys that start with a specific prefix.

**Parameters:**
- `prefix` (str): The prefix to check for.
- `d` (dict): The dictionary to group.

**Returns:**
- `tuple`: Two dictionaries split based on the prefix condition.

---

## groupby_prefix_and_trim(prefix, d)
Group dictionary items by keys that start with a specific prefix and remove the prefix.

**Parameters:**
- `prefix` (str): The prefix to check for.
- `d` (dict): The dictionary to group.

**Returns:**
- `tuple`: Dictionary with the prefix removed and another dictionary with remaining items.

---

## divisible_by(num, den)
Check if num is divisible by den.

**Parameters:**
- `num` (int): The number to check.
- `den` (int): The divisor.

**Returns:**
- `bool`: True if num is divisible by den, False otherwise.

---

## top_p(logits, thres=0.9)
Select top values from logits based on cumulative probability threshold.

**Parameters:**
- `logits` (torch.Tensor): Input logits.
- `thres` (float): Cumulative probability threshold.

**Returns:**
- `torch.Tensor`: Logits with top values selected based on threshold.

---

## top_k(logits, thres=0.9)
Select top values from logits based on threshold.

**Parameters:**
- `logits` (torch.Tensor): Input logits.
- `thres` (float): Probability threshold.

**Returns:**
- `torch.Tensor`: Logits with top values selected based on threshold.

---

## top_a(logits, min_p_pow=2.0, min_p_ratio=0.02)
Apply top-a selection to logits.

**Parameters:**
- `logits` (torch.Tensor): Input logits.
- `min_p_pow` (float): Minimum probability power.
- `min_p_ratio` (float): Minimum probability ratio.

**Returns:**
- `torch.Tensor`: Logits after top-a selection.

---

## log(t, eps=1e-20)
Compute the natural logarithm of a tensor.

**Parameters:**
- `t` (torch.Tensor): Input tensor.
- `eps` (float): Epsilon to avoid zero division.

**Returns:**
- `torch.Tensor`: Natural logarithm of the input tensor.

---

## gumbel_noise(t)
Generate Gumbel noise.

**Parameters:**
- `t` (torch.Tensor): Input tensor.

**Returns:**
- `torch.Tensor`: Gumbel noise tensor.

---

## gumnel_sample(t, temperature=1., dim=-1)
Sample from a tensor using Gumbel-softmax.

**Parameters:**
- `t` (torch.Tensor): Input tensor.
- `temperature` (float): Gumbel-softmax temperature.
- `dim` (int): Dimension along which to sample.

**Returns:**
- `torch.Tensor`: Sampled tensor using Gumbel-softmax.

---

## ContrastiveTopK(nn.Module)
Contrastive Top-k module for score calculation.

**Parameters:**
- `alpha` (float): Alpha parameter.
- `k` (int): K parameter.

**Returns:**
- `torch.Tensor`: Scores calculated using Contrastive Top-k.

---

## print_num_params(model, accelerator: Accelerator)
Print the number of parameters in a model.

**Parameters:**
- `model` (torch.nn.Module): The model to count parameters for.
- `accelerator` (Accelerator): Accelerator object for printing.

---

## Block(nn.Module)
Basic block class.

**Parameters:**
- `dim` (int): Input dimension.
- `dim_out` (int): Output dimension.
- `groups` (int): Number of groups for normalization.

---

## ResnetBlock(nn.Module)
ResNet

 block class.

**Parameters:**
- `dim` (int): Input dimension.
- `dim_out` (int): Output dimension.
- `time_emb_dim` (int): Time embedding dimension (optional).
- `groups` (int): Number of groups for normalization.

---

## load_model(path)
Load a model from a file.

**Parameters:**
- `path` (str): Path to the file containing the model.

**Returns:**
- `torch.nn.Module`: The loaded model.

---

## CHANNELS_TO_MODE
Mapping of channels to image modes.

---

## seek_all_images(img, channels=3)
Seek and yield all images in a GIF.

**Parameters:**
- `img`: GIF image.
- `channels` (int): Number of color channels.

**Yields:**
- PIL.Image: Images from the GIF.

---

## video_tensor_to_gift(tensor, path, duration=120, loop=0, optimize=True)
Convert a tensor of video frames to a GIF.

**Parameters:**
- `tensor` (torch.Tensor): Video tensor.
- `path` (str): Output GIF path.
- `duration` (int): Duration of each frame.
- `loop` (int): Number of loops.
- `optimize` (bool): Optimize GIF.

**Returns:**
- list: List of PIL.Image objects.

---

## gif_to_tensor(path, channels=3, transform=T.ToTensor())
Convert a GIF to a tensor of video frames.

**Parameters:**
- `path` (str): Input GIF path.
- `channels` (int): Number of color channels.
- `transform` (torchvision.transforms): Transform to apply.

**Returns:**
- torch.Tensor: Video tensor.

---

## identity(t, *args, **kwargs)
Identity function that returns the input tensor.

**Parameters:**
- `t` (torch.Tensor): Input tensor.
- `args` (list): Additional positional arguments.
- `kwargs` (dict): Additional keyword arguments.

**Returns:**
- torch.Tensor: Input tensor.

---

## normalize_img(t)
Normalize image tensor to the range [-1, 1].

**Parameters:**
- `t` (torch.Tensor): Input image tensor.

**Returns:**
- torch.Tensor: Normalized image tensor.

---

## unnormalize_img(t)
Unnormalize image tensor from the range [-1, 1] to [0, 1].

**Parameters:**
- `t` (torch.Tensor): Input image tensor.

**Returns:**
- torch.Tensor: Unnormalized image tensor.

---

## cast_num_frames(t, frames)
Cast the number of frames in a tensor.

**Parameters:**
- `t` (torch.Tensor): Input tensor.
- `frames` (int): Number of frames.

**Returns:**
- torch.Tensor: Tensor with specified number of frames.

---

## max_neg_values(tensor)
Get a tensor filled with maximum negative values.

**Parameters:**
- `tensor` (torch.Tensor): Input tensor.

**Returns:**
- torch.Tensor: Tensor filled with maximum negative values.

---

## l2norm(t, groups=1)
Apply L2 normalization along specified dimension.

**Parameters:**
- `t` (torch.Tensor): Input tensor.
- `groups` (int): Number of groups.

**Returns:**
- torch.Tensor: L2 normalized tensor.

---

## pad_at_dim(t, pad, dim=-1, value=0.)
Pad a tensor along a specified dimension.

**Parameters:**
- `t` (torch.Tensor): Input tensor.
- `pad` (tuple): Padding to apply.
- `dim` (int): Dimension to pad along.
- `value` (float): Value for padding.

**Returns:**
- torch.Tensor: Padded tensor.

---

## or_reduce(masks)
Perform element-wise OR reduction on masks.

**Parameters:**
- `masks` (list): List of masks (torch.Tensor).

**Returns:**
- torch.Tensor: Result of element-wise OR reduction.

---

## Residual(nn.Module)
Residual module for adding the input tensor to the output of a function.

**Parameters:**
- `fn` (function): The function to wrap.

**Returns:**
- torch.Tensor: Output tensor.

---

## SinusoidalPosEmb(nn.Module)
Sinusoidal positional embedding module.

**Parameters:**
- `dim` (int): Dimension of the embedding.

**Returns:**
- torch.Tensor: Sinusoidal positional embeddings.

---

## upsample(dim)
Upsample module for increasing resolution.

**Parameters:**
- `dim` (int): Dimension of the input.

**Returns:**
- torch.nn.Module: Upsample module.

---

## downsample(dim)
Downsample module for decreasing resolution.

**Parameters:**
- `dim` (int): Dimension of the input.

**Returns:**
- torch.nn.Module: Downsample module.

---

## LayerNorm(nn.Module)
Layer normalization module.

**Parameters:**
- `dim` (int): Dimension of the input.
- `eps` (float): Epsilon value for stability.

---

## PreNorm(nn.Module)
Pre-normalization module.

**Parameters:**
- `dim` (int): Dimension of the input.
- `fn` (function): The function to wrap.

**Returns:**
- torch.Tensor: Output tensor.

---

## cosine_beta_schedule(timesteps, s=0.008)
Generate a cosine beta schedule.

**Parameters:**
- `timesteps` (int): Number of timesteps.
- `s` (float): Scaling factor.

**Returns:**
- torch.Tensor: Cosine beta schedule.

---

## Normalize(nn.Module)
Normalization module to apply L2 normalization.

**Parameters:**
- `dim` (int): Dimension along which to normalize.

**Returns:**
- torch.Tensor: Normalized tensor.

---

## LearnableLogitScaling(nn.Module)
Learnable logit scaling module.

**Parameters:**
- `logit_scale_init` (float): Initial logit scaling value.
- `learnable` (bool): Whether the logit scaling is learnable.
- `max_logit_scale` (float): Maximum logit scaling value.

---

## EinOpsRearrange(nn.Module)
EinOps-based rearrange module.

**Parameters:**
- `rearrange_expr` (str): EinOps rearrange expression.

**Returns:**
- torch.Tensor: Rearranged tensor.

---

## cast_if_src_dtype(tensor, src_dtype, tgt_dtype)
Cast a tensor to a target dtype if the source dtype matches.

**Parameters:**
- `tensor` (torch.Tensor): Input tensor.
- `src_dtype` (torch.dtype): Source dtype to check.
- `tgt_dtype` (torch.dtype): Target dtype to cast to.

**Returns:**
- torch.Tensor: Casted tensor.

---

## SelectElements(nn.Module)
Select elements from a tensor using given indices.

**Parameters:**
- `index`: Indices to select.

**Returns:**
- torch.Tensor: Selected elements.

---

## SelectEOSAndProject(nn.Module)
Select end-of-sequence (EOS) elements from a tensor and apply projection.

**Parameters:**
- `proj` (nn.Module): Projection module.

**Returns:**
- torch.Tensor: Projected tensor.

---

## get_sinusoid_encoding_table(n_position, d_hid)
Generate sinusoidal positional encoding table.

**Parameters:**
- `n_position` (int): Number of positions.
- `d_hid` (int): Hidden

 dimension.

**Returns:**
- torch.Tensor: Sinusoidal positional encoding table.

---

## interpolate_pos_encoding_2d(target_spatial_size, pos_embed)
Interpolate 2D positional embeddings to a target spatial size.

**Parameters:**
- `target_spatial_size` (int): Target spatial size.
- `pos_embed` (torch.Tensor): Positional embeddings.

**Returns:**
- torch.Tensor: Interpolated positional embeddings.