import re
from functools import wraps
from einops import rearrange, reduce, repeat


def check_shape(tensor, pattern, **kwargs):
    return rearrange(tensor, f"{pattern} -> {pattern}", **kwargs)


# Do many ops on a list of tensors
def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)

    return inner


# Do einops with unflattening of named dimensions
# (...flatenned) -> ...flattened


def _with_anon_dims(fn):
    @wraps(fn)
    def inner(tensor, pattern, **kwargs):
        regex = r"(\.\.\.[a-zA-Z]+)"
        matches = re.findall(regex, pattern)

        def get_anon_dim_name(t):
            return t.lstrip("...")

        dim_prefixes = tuple(map(get_anon_dim_name, matches))

        update_kwargs_dict = dict()

        for prefix in dim_prefixes:
            assert (
                prefix in kwargs
            ), f"dimension list {prefix} not found in kwargs"
            dim_list = kwargs[prefix]
            assert isinstance(
                dim_list, (list, tuple)
            ), f"Dimension list {prefix} needs to be a tuple of list"
            dim_names = list(
                map(lambda ind: f"{prefix}{ind}", range(len(dim_list)))
            )
            update_kwargs_dict[prefix] = dict(zip(dim_names, dim_list))

        def sub_with_anon_dims(t):
            dim_name_prefix = get_anon_dim_name(t.groups()[0])
            return "".join(update_kwargs_dict[dim_name_prefix].keys())

        pattern_new = re.sub(regex, sub_with_anon_dims, pattern)
        return fn(tensor, pattern_new, **kwargs)

    return inner


rearrange_many = _many(rearrange)
repeat_many = _many(repeat)
reduce_many = _many(reduce)

rearrange_with_anon_dims = _with_anon_dims(rearrange)
repeat_with_anon_dims = _with_anon_dims(repeat)
reduce_with_anon_dims = _with_anon_dims(reduce)
