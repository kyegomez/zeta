import math
from functools import partial, wraps
from math import ceil

import einops
import numpy as np
import torch
import torch.functional as F
import torch.nn as nn
from accelerate import Accelerator
from einops import rearrange
from PIL import Image
from torchvision import transforms as T


def exists(val):
    """
    Check if the value is not None.

    Args:
        val: The value to check.

    Returns:
        bool: True if value exists (is not None), False otherwise.
    """
    return val is not None


def default(val, d):
    """
    Return the value if it exists, otherwise return a default value.

    Args:
        val: The value to check.
        d: The default value to return if val is None.

    Returns:
        The value if it exists, otherwise the default value.
    """
    return val if exists(val) else d


def once(fn):
    """
    Decorator to ensure the function is only called once.

    Args:
        fn (function): The function to wrap.

    Returns:
        function: The wrapped function.
    """
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


def eval_decorator(fn):
    """
    Decorator to ensure a method switches to eval mode before execution
    and returns to its original mode afterwards. For torch.nn.Module objects.

    Args:
        fn (function): The function to wrap.

    Returns:
        function: The wrapped function.
    """

    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner


def cast_tuple(val, depth):
    """
    Cast a value to a tuple of a specific depth.

    Args:
        val: Value to be cast.
        depth (int): Depth of the tuple.

    Returns:
        tuple: Tuple of the given depth with repeated val.
    """
    return val if isinstance(val, tuple) else (val,) * depth


def maybe(fn):
    """
    Decorator that calls a function if the first argument exists.

    Args:
        fn (function): The function to wrap.

    Returns:
        function: The wrapped function.
    """

    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)

    return inner


class always:
    """
    Class that always returns a specified value when called.
    """

    def __init__(self, val):
        """
        Initialize the always class with a value.

        Args:
            val: The value to always return.
        """
        self.val = val

    def __call__(self, *args, **kwargs):
        """
        Return the specified value.

        Returns:
            The specified value.
        """
        return self.val


class not_equals:
    """
    Class that checks if a value does not equal the specified value.
    """

    def __init__(self, val):
        """
        Initialize with a value.

        Args:
            val: The value to compare against.
        """
        self.val = val

    def __call__(self, x, *args, **kwargs):
        """
        Compare the input x with the specified value.

        Returns:
            bool: True if x is not equal to the specified value, False otherwise.
        """
        return x != self.val


class equals:
    """
    Class that checks if a value equals the specified value.
    """

    def __init__(self, val):
        """
        Initialize with a value.

        Args:
            val: The value to compare against.
        """
        self.val = val

    def __call__(self, x, *args, **kwargs):
        """
        Compare the input x with the specified value.

        Returns:
            bool: True if x is equal to the specified value, False otherwise.
        """
        return x == self.val


def init_zero_(layer):
    """
    Initialize the weights and bias of a torch layer to zero.

    Args:
        layer (torch.nn.Module): The layer to initialize.
    """
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)


def pick_and_pop(keys, d):
    """
    Remove and return values from a dictionary based on provided keys.

    Args:
        keys (list): List of keys to remove from the dictionary.
        d (dict): The dictionary to pick from.

    Returns:
        dict: A dictionary with the specified keys and their values.
    """
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))


def group_dict_by_key(cond, d):
    """
    Group dictionary keys based on a condition.

    Args:
        cond (function): Condition to split dictionary.
        d (dict): The dictionary to group.

    Returns:
        tuple: Two dictionaries split based on the condition.
    """
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    """
    Check if a string begins with a specific prefix.

    Args:
        prefix (str): The prefix to check for.
        str (str): The string to check.

    Returns:
        bool: True if string starts with prefix, False otherwise.
    """
    return str.startswith(prefix)


def group_by_key_prefix(prefix, d):
    """
    Group dictionary items by keys that start with a specific prefix.

    Args:
        prefix (str): The prefix to check for.
        d (dict): The dictionary to group.

    Returns:
        tuple: Two dictionaries split based on the prefix condition.
    """
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(prefix, d):
    """
    Group dictionary items by keys that start with a specific prefix and remove the prefix.

    Args:
        prefix (str): The prefix to check for.
        d (dict): The dictionary to group.

    Returns:
        tuple: Dictionary with the prefix removed and another dictionary with remaining items.
    """
    kwargs_with_prefix, kwargs = group_dict_by_key(
        partial(string_begins_with, prefix), d
    )
    kwargs_without_prefix = dict(
        map(
            lambda x: (x[0][len(prefix) :], x[1]),
            tuple(kwargs_with_prefix.items()),
        )
    )
    return kwargs_without_prefix, kwargs


def divisible_by(num, den):
    return (num % den) == 0


def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def top_k(logits, thres=0.9):
    k = ceil((1 - thres) * logits.shape[-1])
    (
        val,
        ind,
    ) = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


def top_a(logits, min_p_pow=2.0, min_p_ratio=0.02):
    probs = F.softmax(logits, dim=-1)
    limit = torch.pow(torch.max(probs), min_p_pow) * min_p_ratio

    logits[probs < limit] = float("-inf")
    logits[probs >= limit] = 1
    return logits


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumnel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


class ContrastiveTopK(nn.Module):
    def __init__(self, alpha, k):
        super(ContrastiveTopK, self).__init__()
        self.alpha = alpha
        self.k = k

    def top_k(self, logits):
        k = ceil((1 - self.alpha) * logits.shape[-1])
        val, ind = torch.topk(logits, k)

        probs = torch.full_like(logits, float("-inf"))
        probs.scatter_(1, ind, val)

        return probs

    def forward(self, logits_exp, logits_ama):
        logits_exp_topk = self.top_k(logits_exp)
        logits_ama_topk = self.top_k(logits_ama)

        # probabilities
        p_exp = F.softmax(logits_exp_topk, dim=-1)
        p_ama = F.softmax(logits_ama_topk, dim=-1)

        # mask
        _, ind = torch.topk(p_exp, self.k)
        mask = torch.zeros_like(p_exp)
        mask.scatter_(1, ind, p_exp[ind] >= self.alpha * p_exp[ind[-1]])

        # scores
        scores = torch.where(
            mask.bool(),
            torch.log(p_exp / (p_ama + 1e-8)),
            torch.tensor(-float("inf")),
        )

        return scores


# alpha = 0.5


def print_num_params(model, accelerator: Accelerator):
    # n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Number of parameters in model: {n_params}")


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + 1

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = (
            nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), "time_emb must be passed in"
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


def load_model(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location=torch.device("cpu"))


CHANNELS_TO_MODE = {1: "L", 3: "RGB", 4: "RGBA"}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f"channels {channels} invalid"
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1


# tensor of shape (channels, frames, height, width) -> GIF
def video_tensor_to_gift(tensor, path, duration=120, loop=0, optimize=True):
    images = map(T.ToPilImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(
        path,
        save_all=True,
        appeqnd_images=rest_imgs,
        duration=duration,
        loop=loop,
        optimize=optimize,
    )
    return images


# gif -> (channels, frame, height, width) tensor
def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, chanels=channels)))
    return torch.stack(tensors, dim=1)


def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(t):
    return (t + 1) * 0.5


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


def max_neg_values(tensor):
    return -torch.info(tensor.dtype).max


def l2norm(t, groups=1):
    t = rearrange(t, "... (g d) -> ... g d", g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, "... g d -> ... (g d)")


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]

        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.vart(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = (
        torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)


class LearnableLogitScaling(nn.Module):
    def __init__(
        self,
        logit_scale_init: float = 1 / 0.07,
        learnable: bool = True,
        max_logit_scale: float = 100,
    ) -> None:
        super().__init__()
        self.max_logit_scale = max_logit_scale
        self.logit_scale_init = logit_scale_init
        self.learnable = learnable

        log_logit_scale = torch.ones([]) * np.log(self.logit_scale_init)
        if learnable:
            self.log_logit_scale = nn.Parameter(log_logit_scale)
        else:
            self.register_bufffer("log_logit_scale", log_logit_scale)

    def forward(self, x):
        return torch.clip(self.logit_scale.exp(), max=self.max_logit_scale) * x

    def extra_repr(self):
        st = (
            f"logit_scale_init={self.logit_scale_init},"
            f" learnable={self.learnable},"
            f"max_logit_scale={self.max_logit_scale}"
        )
        return st


class EinOpsRearrange(nn.Module):
    def __init__(self, rearrange_expr: str, **kwargs) -> None:
        super().__init__()
        self.rearrange_expr = rearrange_expr
        self.kwargs = kwargs

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        return einops.rearrange(x, self.rearrange_expr, **self.kwargs)


def cast_if_src_dtype(
    tensor: torch.Tensor, src_dtype: torch.dtype, tgt_dtype: torch.dtype
):
    updated = False
    if tensor.dtype == src_dtype:
        tensor = tensor.to(dtype=tgt_dtype)
        updated = True
    return tensor, updated


class SelectElements(nn.Module):
    def __init__(self, index) -> None:
        super().__init__()
        self.index = index

    def forward(self, x):
        assert x.ndim >= 3
        return x[:, self.index, ...]


class SelectEOSAndProject(nn.Module):
    def __init__(self, proj: nn.Module) -> None:
        super().__init__()
        self.proj = proj

    def forward(self, x, seq_len):
        assert x.ndim == 3
        x = x[torch.arange(x.shape[0]), seq_len]
        x = self.proj(x)
        return x


##################
def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 21
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def interpolate_pos_encoding_2d(target_spatial_size, pos_embed):
    N = pos_embed.shape[1]
    if N == target_spatial_size:
        return pos_embed
    dim = pos_embed.shape[-1]
    pos_embed, updated = cast_if_src_dtype(
        pos_embed, torch.bfloat16, torch.float32
    )
    pos_embed = nn.functional.interpolate(
        pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
            0, 3, 1, 2
        ),
        scale_factor=math.sqrt(target_spatial_size / N),
        mode="bicubic",
    )
    if updated:
        pos_embed, _ = cast_if_src_dtype(
            pos_embed, torch.float32, torch.bfloat16
        )
    pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return pos_embed


#############

# def init_bert_params(module):
#     def normal_(data):
#         data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

#     if isinstance(module, nn.Linear):
#         normal_(module.weight.data)
#         if module.bias is not None:
#             module.bias.data.zero_()
#     if isinstance(module, nn.Embedding):
#         normal_(module.weight.data)
#         if module.padding_idx is not None:
#             module.weight.data[module.padding_idx].zero_()
#     if isinstance(module, MultiheadAttention):
#         if isinstance(module.q_proj, MultiwayNetwork):
#             normal_(module.q_proj.A.weight.data)
#             normal_(module.q_proj.B.weight.data)
#             normal_(module.k_proj.A.weight.data)
#             normal_(module.k_proj.B.weight.data)
#             normal_(module.v_proj.A.weight.data)
#             normal_(module.v_proj.B.weight.data)
#         else:
#             normal_(module.q_proj.weight.data)
#             normal_(module.k_proj.weight.data)
#             normal_(module.v_proj.weight.data)


######
def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value=value)


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)

    tensors = [
        padded_x[:, ind : (ind + t), ...]
        for ind in range(forward + backward + 1)
    ]
    return torch.cat(tensors, dim=dim)


####


def is_power_of_two(n):
    return math.log2(n).is_integer()


def all_unique(arr):
    return len(set(arr)) == len(arr)


def apply_fns(fns, tensors):
    return [fn(tensors) for fn, tensor in zip(fns, tensors)]


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)
