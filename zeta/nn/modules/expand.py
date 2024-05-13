from einops import repeat

def expand(*args, **kwargs):
    return repeat(*args, **kwargs)