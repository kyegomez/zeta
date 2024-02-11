from einops import rearrange, pack, unpack
from functools import wraps


def exists(val):
    return val is not None


def pack_one(x, pattern):
    return pack([x], pattern)


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


def compact_values(d: dict):
    return {k: v for k, v in d.items() if exists(v)}


def image_or_video_to_time(fn):
    """
    Decorator function that converts the input tensor from image or video format to time format.

    Args:
        fn: The function to be decorated.

    Returns:
        The decorated function.
    """

    @wraps(fn)
    def inner(self, x, batch_size=None, **kwargs):
        is_video = x.ndim == 5

        if is_video:
            batch_size = x.shape[0]
            x = rearrange(x, "b c t h w -> b h w c t")
        else:
            assert exists(batch_size) or exists(self.time_dim)
            rearrange_kwargs = dict(b=batch_size, t=self.time_dim)
            x = rearrange(
                x,
                "(b t) c h w -> b h w c t",
                **compact_values(rearrange_kwargs),
            )

        x, ps = pack_one(x, "* c t")

        x = fn(self, x, **kwargs)

        x = unpack_one(x, ps, "* c t")

        if is_video:
            x = rearrange(x, "b h w c t -> b c t h w")
        else:
            x = rearrange(x, "b h w c t -> (b t) c h w")

        return x

    return inner
