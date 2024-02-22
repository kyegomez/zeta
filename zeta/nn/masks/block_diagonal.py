import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mask(self, n, device=device):
    if self.mask is not None and self.mask.shape[-1] >= n:
        return self.mask[:n, :n]

    if self.mask is None:
        print("computing mask..")

    mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
    k = 0
    segment_lengths = [4, 8, 16]
    dilation_rates = [1, 2, 4]
    # segment_lengths = [2048, 4096, 8192, 16384, 32768]
    # dilation_rates = [1, 2, 4, 6, 12]
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            will_mask = True
            for segment_length, dilation_rate in zip(
                segment_lengths, dilation_rates
            ):
                if (
                    np.floor(i / segment_length) == np.floor(j / segment_length)
                    and i % dilation_rate == 0
                    and j % dilation_rate == 0
                ):
                    will_mask = False
            if will_mask:
                mask[i][j] = True
            k += 1
    self.register_buffer("mask", mask, persistent=False)
    self.mask = mask
    return mask


x = torch.randn(1, 3, 32, 32)

model = get_mask(n=x)
print(model)
