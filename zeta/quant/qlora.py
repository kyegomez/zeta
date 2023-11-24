from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from tqdm import tqdm
import math

bnb_available = False


def get_block_absmax(
    inpt_tensor: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Iterate through a flattened tensor getting the absmax scalers for each block

    Args:
        inpt_tensor: Input tensor to get scalers for
        block_size: Block size for the scanning window
    Returns:
        torch.Tensor: Tensor of scalers for each block
    """
    assert inpt_tensor.dim() == 1, "Input tensor must be flattened"
    assert (inpt_tensor.numel() % block_size) == 0, (
        "Input tensor must be divisible by block size, got"
        f" {inpt_tensor.numel()} and {block_size}"
    )

    n_blocks = inpt_tensor.numel() // block_size
    blocks = inpt_tensor.view(n_blocks, block_size)
    block_scalers = blocks.abs().max(dim=1).values
    return block_scalers


class NF4Tensor:
    """NF4Tensor class for converting a weight to the QLoRA NF4 format"""

    @classmethod
    @torch.no_grad()
    def from_tensor(
        cls,
        inpt_tensor: torch.Tensor,
        block_size: int = 64,
        scaler_block_size: int = 256,
    ):
        assert inpt_tensor.dtype == torch.bfloat16
        assert (
            inpt_tensor.numel() % block_size == 0
        ), "Input tensor must be divisible by block size"
        assert (
            inpt_tensor.dtype == torch.bfloat16
        ), "Input tensor must be bfloat16"
        device = inpt_tensor.device
        # Cache the tensor on the class def
        nf4 = torch.tensor(
            [
                -1.0000,
                -0.6962,
                -0.5251,
                -0.3949,
                -0.2844,
                -0.1848,
                -0.0911,
                0.0000,
                0.0796,
                0.1609,
                0.2461,
                0.3379,
                0.4407,
                0.5626,
                0.7230,
                1.0000,
            ],
            device=device,
            dtype=torch.bfloat16,
        )
        n_blocks = inpt_tensor.numel() // block_size
        # Double quantization
        (
            quantized_scalers,
            quantization_factor,
            scaler_mean,
        ) = cls.double_quantize_scalers(
            inpt_tensor.flatten(), block_size, scaler_block_size
        )
        quantized_data = cls.convert_to_norm_float_weight(
            inpt_tensor, n_blocks, block_size, nf4
        )
        original_shape = inpt_tensor.shape
        return cls(
            block_size,
            n_blocks,
            scaler_block_size,
            quantized_scalers,
            quantization_factor,
            scaler_mean,
            quantized_data,
            original_shape,
            nf4=nf4,
        )

    def __init__(
        self,
        block_size: int,
        n_blocks: int,
        scaler_block_size: int,
        quantized_scalers: torch.Tensor,
        quantization_factor: torch.Tensor,
        scaler_mean: torch.Tensor,
        quantized_data: torch.Tensor,
        original_shape: torch.Size,
        nf4: torch.Tensor,
    ):
        """Initialize the NF4Tensor class"""
        self.device = quantized_data.device
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.scaler_block_size = scaler_block_size
        self.quantized_scalers = quantized_scalers
        self.quantization_factor = quantization_factor
        self.scaler_mean = scaler_mean
        self.quantized_data = quantized_data
        self.original_shape = original_shape
        self.nf4 = nf4

    @staticmethod
    def double_quantize_scalers(
        inpt_tensor: torch.Tensor,
        block_size: int,
        scaler_block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used to achieve the double quantization of the scalers
        We take the input tensor first calculate the absmax quantization factors for each block.
        We then find the mean of our positive absmax scalers. We subtract this mean from the scalers
        And then we calculate the absmax quantization factors for each block again. We then quantize the scalers to int8.

        Args:
            inpt_tensor: Input tensor to convert to QLoRA format, typically a weight tensor

        Returns:
            torch.Tensor: Tensor of per_block quantization factors stored in int8 format
                size: (n_blocks)
            torch.Tensor: Tensor of per_scaler_block quantization factors stored in int16 format
                size: (n_scaler_blocks)
        """
        assert inpt_tensor.dim() == 1, "Input tensor must be flattened"
        assert (inpt_tensor.numel() % scaler_block_size) == 0, (
            "Input tensor must be divisible by block size, got"
            f" {inpt_tensor.numel()} and {scaler_block_size}"
        )

        # First round of quantization
        # Produces: A tensor of size (n_blocks) of inpt_tensor.dtype
        scalers_1 = get_block_absmax(inpt_tensor, block_size)
        scalers_1_mean = scalers_1.mean()
        scalers_1 = scalers_1 - scalers_1_mean
        # Second round of quantization
        assert (
            scalers_1.numel() % scaler_block_size == 0
        ), "Number of scalers must be divisible by scaler block size"
        n_scaler_blocks = scalers_1.numel() // scaler_block_size
        scaler_blocks = scalers_1.view(n_scaler_blocks, scaler_block_size)

        scaler_absmax = get_block_absmax(scalers_1, scaler_block_size)
        scaler_absmax = scaler_absmax.unsqueeze(-1).expand(
            n_scaler_blocks, scaler_block_size
        )

        quantization_factor = 256 / (2 * scaler_absmax)
        quantized_scaler_blocks = scaler_blocks * quantization_factor
        quantized_scaler_blocks = quantized_scaler_blocks.round()
        quantized_scaler_blocks = quantized_scaler_blocks.clamp(-128, 127)

        # This is needed to make sure that quantization_factor remains a repeated view of n_scaler_blocks
        # For some reason the 127/scaler_absmax realizes n_scaler entries when only n_scaler_blocks are needed
        # The following will grab the first entry for the n_scaler_blocks which is the same across the scaler_block_size
        quantization_factor = quantization_factor[:, 0]

        return (
            quantized_scaler_blocks.flatten().to(torch.int8),
            quantization_factor.view(n_scaler_blocks),
            scalers_1_mean,
        )

    def dequantize_scalers(
        self,
        inpt_tensor: torch.Tensor,
        quantization_factor: torch.Tensor,
        scaler_block_size: int,
    ) -> torch.Tensor:
        """Used to unpack the double quantized scalers

        Args;
            inpt_tensor: Input tensor to convert to QLoRA format this is the quantized scalers in int8 format
            quantization_factor: Tensor of per_scaler_block quantization factors stored in inpt_weight.dtype
                size: (n_scaler_blocks)
            scaler_block_size: Scaler block size to use for double quantization.

        """
        assert inpt_tensor.dim() == 1, "Input tensor must be flattened"
        assert (inpt_tensor.numel() % scaler_block_size) == 0, (
            "Input tensor must be divisible by block size, got"
            f" {inpt_tensor.numel()} and {scaler_block_size}"
        )
        n_scaler_blocks = inpt_tensor.numel() // scaler_block_size
        inpt_tensor = inpt_tensor.view(n_scaler_blocks, scaler_block_size)
        dequantized = (
            inpt_tensor / quantization_factor.unsqueeze(-1)
        ).flatten().to(torch.bfloat16) + self.scaler_mean
        return dequantized

    @staticmethod
    def convert_to_norm_float_weight(
        inpt_tensor: torch.Tensor,
        n_blocks: int,
        block_size: int,
        nf4: torch.tensor,
    ) -> torch.Tensor:
        """Convert a tensor to the normalized float weight format"""
        flattened_tensor = inpt_tensor.flatten()
        #  Since we are using uint8 we will encode 2 entries per byte
        numel = inpt_tensor.numel()
        assert numel % 2 == 0, (
            "Number of elements must be even just to not have to think about"
            " the end"
        )
        # Reshape the flattened tensor into blocks of size self.block_size
        blocks = flattened_tensor.view(n_blocks, block_size)

        # Scale the blocks
        scalers = get_block_absmax(inpt_tensor.flatten(), block_size)
        scales = scalers.unsqueeze(-1).expand(n_blocks, block_size)
        scaled_blocks = blocks / scales

        # Returns a flattened tensor with each element quantized to nf4 index
        # The weird behavior comes here with how qlora vs bnb break nf4 ties.
        # Since we ust torch.min(nf4 - inpt/scale) we will always pick the smallest index
        # While bnb appears to be pick the larger index when breaking ties
        # ACTUALLYYY I think that what ever op bnb is using to get the nearest NF4 value
        # Is not consistent with torch.round. Example: input 1.1016 with abs max
        # scale of 2.2821 will get mapped to 1.25 while mine will get mapped to 0.9570
        # The difference for mine is 0.1445 and for bnb 0.1484
        quantized_blocks = NF4Tensor.quantize_tensor_nearest(
            scaled_blocks.flatten(), nf4
        )

        # Combine the quantized elements into uint8 values
        combined_blocks = quantized_blocks[::2] << 4 | quantized_blocks[1::2]

        return combined_blocks.to(torch.uint8)

    def get_original_weight(self) -> torch.Tensor:
        """Get the original weight from the normalized float weight format"""
        # since we are using uint8 we will decode 2 entries per byte
        # Shift elements down 4 and select out the bottom 4 bits
        first_elements = (self.quantized_data >> 4).to(torch.long)
        second_elements = (self.quantized_data & 0b1111).to(torch.long)

        # Dequantize every element
        dequantized_first = self.dequantize(first_elements, self.nf4)
        dequantized_second = self.dequantize(second_elements, self.nf4)

        # Build up matrix of scalers repeated for each element in the block
        # Since first and second elements make up a full block, so
        # we expand out to half the size of the full block
        scalers = self.dequantize_scalers(
            self.quantized_scalers,
            self.quantization_factor,
            self.scaler_block_size,
        )
        repeated = scalers.unsqueeze(-1).expand(
            scalers.size(0), self.block_size // 2
        )

        scaled_first = dequantized_first * repeated.flatten()
        scaled_second = dequantized_second * repeated.flatten()

        # Flip them to be vertical and them stack them together horizontally
        # Upon flattening this will interleave the elements
        scaled_first = scaled_first.unsqueeze(-1).transpose(0, 1)
        scaled_second = scaled_second.unsqueeze(-1).transpose(0, 1)
        return torch.stack([scaled_first, scaled_second], dim=-1).reshape(
            self.original_shape
        )

    @staticmethod
    def quantize_tensor_nearest(
        value: torch.float16, nf4: torch.Tensor
    ) -> torch.Tensor:
        """Quantize a float16 tensor to nf4 format to nearest and not rounded up"""
        value = value.unsqueeze(-1)  # (numel, 1)
        # Compare the value tensor with the nf4 tensor element-wise
        diff = (value - nf4).abs()
        # BnB appears to break ties by choosing the larger nf4 value
        closest_nf4 = diff.min(dim=-1).indices
        return closest_nf4

    @staticmethod
    def dequantize(value: torch.Tensor, nf4: torch.Tensor) -> torch.Tensor:
        """Dequantize a nf4 value to float16 format"""
        # return nf4.index_select(0, value)
        return nf4[value]

    def unpack(
        self,
    ) -> Tuple[
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Size,
    ]:
        return (
            self.block_size,
            self.n_blocks,
            self.scaler_block_size,
            self.quantized_scalers,
            self.quantization_factor,
            self.scaler_mean,
            self.quantized_data,
            self.original_shape,
        )

    def __repr__(self):
        return (
            f"Quantized Data: {self.quantized_data}\nScalers:"
            f" {self.quantized_scalers}\n"
        )

    def __str__(self):
        return f"NF4Tensor({self.original_shape}, {self.block_size})"


class NF4TensorDebug:
    """QLoRA Weight written in a more Debug friendly manner"""

    @staticmethod
    def get_nf4(cached=True) -> torch.Tensor:
        if cached:
            return torch.tensor(
                [
                    -1.0000,
                    -0.6962,
                    -0.5251,
                    -0.3949,
                    -0.2844,
                    -0.1848,
                    -0.0911,
                    0.0000,
                    0.0796,
                    0.1609,
                    0.2461,
                    0.3379,
                    0.4407,
                    0.5626,
                    0.7230,
                    1.0000,
                ]
            )

        offset = 0.9677083
        v1 = norm.ppf(torch.linspace(offset, 0.5, 9)[:-1]).tolist()
        # v2 = [0]*(256-15)
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, 8)[:-1])).tolist()
        # v = v1 + v3 + 0.0
        nkf = torch.tensor(v1 + v3 + [0.0])
        nkf = nkf.sort().values
        nkf /= nkf.max()
        return nkf

    @staticmethod
    def quantize(value: torch.float16, nkf: torch.Tensor) -> torch.Tensor:
        """Quantize a float16 value to nkf format"""
        for i in range(len(nkf)):
            if value <= nkf[i]:
                # print("value", value, "nkf", nkf[i])
                return 0 | i
        return 0 | (len(nkf) - 1)

    @staticmethod
    def quantize_nearest(
        value: torch.float16, nkf: torch.Tensor
    ) -> torch.Tensor:
        closest_index = 0
        closest_diff = abs(nkf[0] - value)
        for i in range(1, len(nkf)):
            diff = abs(nkf[i] - value)
            if diff < closest_diff:
                closest_diff = diff
                closest_index = i
        return 0 | closest_index

    @staticmethod
    def dequantize(value: torch.Tensor, nkf: torch.Tensor) -> torch.Tensor:
        """Dequantize a nkf value to float16 format"""
        # return nkf.index_select(0, value)
        return nkf[value]

    def get_scalers(
        self, inpt_tensor: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        """Iterate through a flattened tensor getting the scalers for each block"""
        flattened_tensor = inpt_tensor.flatten()
        block_scalers = []
        for block_start in range(0, inpt_tensor.numel(), block_size):
            block_end = min(block_start + block_size, inpt_tensor.numel())
            block = flattened_tensor[block_start:block_end]
            block_max = block.abs().max()
            block_scalers.append(block_max)
        return torch.tensor(block_scalers)

    def __init__(self, inpt_tensor: torch.Tensor, block_size=64):
        assert inpt_tensor.dtype == torch.bfloat16
        assert (
            inpt_tensor.numel() % block_size == 0
        ), "Input tensor must be divisible by block size"
        self.block_size = block_size
        self.n_blocks = inpt_tensor.numel() // block_size
        self.scalers = self.get_scalers(inpt_tensor, self.block_size)
        self.norm_float_weight = self.get_norm_float_weight(inpt_tensor.clone())
        self.original_shape = inpt_tensor.shape

    def get_norm_float_weight(self, inpt_tensor: torch.Tensor) -> torch.Tensor:
        nkf = self.get_nf4()
        flattened_tensor = inpt_tensor.flatten()
        #  Since we are using uint8 we will encode 2 entries per byte
        numel = inpt_tensor.numel()
        assert numel % 2 == 0, (
            "Number of elements must be even just to not have to think about"
            " the end"
        )
        quantized_length = numel // 2
        quantized_tensor = torch.zeros(quantized_length, dtype=torch.uint8)
        for i in tqdm(range(len(self.scalers))):
            block_start = i * self.block_size
            block_end = min(
                block_start + self.block_size, flattened_tensor.numel()
            )
            block = flattened_tensor[block_start:block_end]
            # Scale the block
            block /= self.scalers[i]
            # We will iterate over each element in the block and quantize it
            # In groups of 2
            for j in range(0, self.block_size, 2):
                # Combine two bfloat16s via quantization to 4 bit types into a single uint8
                element_1 = self.quantize_nearest(block[j], nkf)
                element_2 = self.quantize_nearest(block[j + 1], nkf)
                combined = element_1 << 4 | element_2
                quantized_tensor[(i * self.block_size // 2) + j // 2] = combined
        return quantized_tensor

    def get_original_weight(self):
        # since we are using uint8 we will decode 2 entries per byte
        nkf = self.get_nf4()
        original_weight = torch.empty(
            2 * (self.norm_float_weight.numel()), dtype=torch.bfloat16
        )
        # Scalers is a proxy for num_blocks
        for i in range(len(self.scalers)):
            block_start = i * self.block_size
            block_end = block_start + self.block_size
            block = original_weight[block_start:block_end]
            for j in range(0, self.block_size, 2):
                combined = self.norm_float_weight[
                    (i * self.block_size // 2) + j // 2
                ]
                # Shift element down 4
                element_1 = combined >> 4
                # Select out the bottom 4 bits
                element_2 = combined & 0b1111
                block[j] = (
                    self.dequantize(element_1.item(), nkf) * self.scalers[i]
                )
                block[j + 1] = (
                    self.dequantize(element_2.item(), nkf) * self.scalers[i]
                )
        return original_weight.reshape(self.original_shape)


class LinearNF4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: NF4Tensor):
        ctx.nf4_weight = weight
        return F.linear(input, weight.get_original_weight())

    @staticmethod
    def backward(ctx, grad_output):
        weight: NF4Tensor = ctx.nf4_weight
        return grad_output @ weight.get_original_weight(), None


def linear_nf4(input: torch.Tensor, weight: NF4Tensor) -> torch.Tensor:
    return LinearNF4.apply(input, weight)


def build_input_weight(embed_dim: int, device: torch.device):
    torch.manual_seed(0)
    input_weight = torch.empty(
        embed_dim, embed_dim, device=device, dtype=torch.bfloat16
    )
    input_weight.normal_(0, 1)
    return input_weight


def build_bitsandbytes_linear(input_weight: torch.Tensor, device: torch.device):
    global bnb
    if "bnb" not in globals():
        import bitsandbytes as bnb
    param = bnb.nn.Params4bit(
        input_weight, requires_grad=False, quant_type="nf4"
    ).cuda(device)
    bnb_linear = bnb.nn.LinearNF4(
        input_weight.size(0), input_weight.size(1), bias=False
    )
    bnb_linear.weight = param
    bnb_linear.to(device)
    return bnb_linear


def get_sample_inputs(
    bsz: int,
    seqlen: int,
    embed_dim: int,
    device: torch.device,
    requires_grad: bool = False,
) -> torch.Tensor:
    sample_input = torch.rand(
        bsz,
        seqlen,
        embed_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=requires_grad,
    )
    sample_input = sample_input.view(bsz * seqlen, embed_dim)
    return sample_input


def get_mlp_weights(
    embed_dim: int, device: torch.dtype = torch.device("cuda:0")
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """These three weights take up
    3 * (embed_dim * n_hidden) * 2 bytes of memory
    i.g. for embed_dim = 4096 and hidden_dim = 11008
    Total memory usage is 270532608 bytes or 0.27 gb
    """
    torch.manual_seed(0)

    def find_multiple(n: int, k: int) -> int:
        if n % k == 0:
            return n
        return n + k - (n % k)

    hidden_dim = 4 * embed_dim
    n_hidden = int(2 * hidden_dim / 3)
    n_hidden = find_multiple(n_hidden, 256)
    weight1 = torch.empty(
        (n_hidden, embed_dim), dtype=torch.bfloat16, device=device
    ).normal_(0, 1)
    weight2 = torch.empty(
        (n_hidden, embed_dim), dtype=torch.bfloat16, device=device
    ).normal_(0, 1)
    weight3 = torch.empty(
        (embed_dim, n_hidden), dtype=torch.bfloat16, device=device
    ).normal_(0, 1)

    return weight1, weight2, weight3


class MLP(nn.Module):
    def __init__(self, weight1, weight2, weight3) -> None:
        super().__init__()
        self.w1, self.w2, self.w3 = weight1, weight2, weight3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(F.linear(x, self.w1)) * F.linear(x, self.w2)
        x = F.linear(x, self.w3)
        return x


class NF4MLP(nn.Module):
    def __init__(self, weight1, weight2, weight3) -> None:
        super().__init__()
        self.w1 = NF4Tensor.from_tensor(weight1)
        self.w2 = NF4Tensor.from_tensor(weight2)
        self.w3 = NF4Tensor.from_tensor(weight3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(linear_nf4(x, self.w1)) * linear_nf4(x, self.w2)
        x = linear_nf4(x, self.w3)
        return x


class BnbQloraMLP(nn.Module):
    def __init__(self, weight1, weight2, weight3, device) -> None:
        super().__init__()
        self.w1 = build_bitsandbytes_linear(weight1, device)
        self.w2 = build_bitsandbytes_linear(weight2, device)
        self.w3 = build_bitsandbytes_linear(weight3, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.w3(x)
        return x


class QloraLinear(nn.Module):
    """
    QloRA Linear Layer

    QloraLinear is a module that performs a linear transformation on the input data.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        weight: weight tensor of shape (out_features, in_features)
        r: number of blocks to use for QLoRA
        lora_alpha: scaling factor for QLoRA
        lora_dropout: dropout to apply to the QLoRA term

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features, in_features). The values are initialized from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where :math:`k = \frac{1}{\text{in_features}}`
        lora_A: the learnable weights of the QLoRA A term of shape
            (r, in_features). The values are initialized from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where :math:`k = \frac{1}{\text{in_features}}`
        lora_B: the learnable weights of the QLoRA B term of shape
            (out_features, r). The values are initialized to zero
        scaling: the scaling factor for the QLoRA term

    Example:
        import torch
        from zeta.quant.qlora import QloraLinear
        # Convert the weight tensor to torch.bfloat16
        weight_bfloat16 = torch.rand(4096, 4096).to(torch.bfloat16)

        # Create the QloraLinear model with the correctly typed weight tensor
        model = QloraLinear(4096, 4096, weight=weight_bfloat16, r=64)

        # Convert the input tensor to torch.bfloat16
        tensor = torch.rand(4096, 4096).to(torch.bfloat16)

        # Perform a forward and backward pass
        out = model(tensor).sum()
        print(out)
        out.backward()


    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        r: int,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.weight = NF4Tensor.from_tensor(weight)
        self.r = r
        self.lora_alpha = lora_alpha
        self.in_features = in_features
        self.out_features = out_features
        self.lora_A = nn.Parameter(weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(weight.new_zeros((out_features, r)))
        self.scaling = self.lora_alpha / self.r

        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = linear_nf4(x, self.weight)
        result2 = (
            result
            + (
                self.lora_dropout(x)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            )
            * self.scaling
        )
        return result2
