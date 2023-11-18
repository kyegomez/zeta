import torch

try:
    from flashfftconv import FlashFFTConv
except ImportError:
    raise ImportError("Please install the flashfftconv package")


class FlashFFTConvWrapper:
    def __init__(self, fft_size, dtype=torch.bfloat16):
        self.flash_fft_conv = FlashFFTConv(fft_size, dtype)

    def __call__(self, x, k):
        return self.flash_fft_conv(x, k)
