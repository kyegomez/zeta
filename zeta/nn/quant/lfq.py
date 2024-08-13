"""
Lookup Free Quantization
Proposed in https://arxiv.org/abs/2310.05737

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.
"""

from collections import namedtuple
from math import ceil, log2

import torch
import torch.nn.functional as F
from einops import pack, rearrange, reduce, unpack
from torch import Tensor, einsum, nn
from torch.nn import Module

# constants

Return = namedtuple("Return", ["quantized", "indices", "entropy_aux_loss"])

LossBreakdown = namedtuple(
    "LossBreakdown", ["per_sample_entropy", "batch_entropy", "commitment"]
)

# helper functions


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# entropy


def log(t, eps=1e-5):
    return t.clamp(min=eps).log()


def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)


# class


class LFQ(Module):
    """
    Initializes the Lookup-Free Quantization (LFQ) module.

    Args:
        dim (int, optional): The input dimension. If not specified, it is calculated based on the codebook size and number of codebooks. Defaults to None.
        codebook_size (int, optional): The size of the codebook. If not specified, it is calculated based on the input dimension. Defaults to None.
        entropy_loss_weight (float, optional): The weight for the entropy loss. Defaults to 0.1.
        commitment_loss_weight (float, optional): The weight for the commitment loss. Defaults to 0.25.
        diversity_gamma (float, optional): The gamma parameter for diversity regularization. Defaults to 1.0.
        straight_through_activation (nn.Module, optional): The activation function to be used during the forward pass. Defaults to nn.Identity().
        num_codebooks (int, optional): The number of codebooks. Defaults to 1.
        keep_num_codebooks_dim (bool, optional): Whether to keep the number of codebooks dimension. Defaults to None.
        codebook_scale (float, optional): The scale factor for the codebook. Defaults to 1.0.

    Examples::
        import torch
        from zeta.nn import LFQ

        # you can specify either dim or codebook_size
        # if both specified, will be validated against each other

        quantizer = LFQ(
            codebook_size = 65536,      # codebook size, must be a power of 2
            dim = 16,                   # this is the input feature dimension, defaults to log2(codebook_size) if not defined
            entropy_loss_weight = 0.1,  # how much weight to place on entropy loss
            diversity_gamma = 1.        # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
        )

        image_feats = torch.randn(1, 16, 32, 32)

        quantized, indices, entropy_aux_loss = quantizer(image_feats)

        # (1, 16, 32, 32), (1, 32, 32), (1,)

        assert image_feats.shape == quantized.shape
        assert (quantized == quantizer.indices_to_codes(indices)).all()
    """

    def __init__(
        self,
        *,
        dim=None,
        codebook_size=None,
        entropy_loss_weight=0.1,
        commitment_loss_weight=0.25,
        diversity_gamma=1.0,
        straight_through_activation=nn.Identity(),
        num_codebooks=1,
        keep_num_codebooks_dim=None,
        codebook_scale=1.0,  # for residual LFQ, codebook scaled down by 2x at each layer
    ):
        super().__init__()

        # some assert validations

        assert exists(dim) or exists(
            codebook_size
        ), "either dim or codebook_size must be specified for LFQ"
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), (
            "your codebook size must be a power of 2 for lookup free"
            f" quantization (suggested {2 ** ceil(log2(codebook_size))})"
        )

        codebook_size = default(codebook_size, lambda: 2**dim)
        codebook_dim = int(log2(codebook_size))

        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = dim != codebook_dims
        self.project_in = (
            nn.Linear(dim, codebook_dims) if has_projections else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_dims, dim) if has_projections else nn.Identity()
        )
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(
            keep_num_codebooks_dim, num_codebooks > 1
        )
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # straight through activation

        self.activation = straight_through_activation

        # entropy aux loss related weights

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # codebook scale

        self.codebook_scale = codebook_scale

        # commitment loss

        self.commitment_loss_weight = commitment_loss_weight

        # for no auxiliary loss, during inference

        self.register_buffer(
            "mask", 2 ** torch.arange(codebook_dim - 1, -1, -1)
        )
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # codes

        all_codes = torch.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)

        self.register_buffer("codebook", codebook, persistent=False)

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self):
        return self.codebook.dtype

    def indices_to_codes(self, indices, project_out=True):
        """Indices to codes.

        Args:
            indices (_type_): _description_
            project_out (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... -> ... 1")

        # indices to codes, which are bits of either -1 or 1

        bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)

        codes = self.bits_to_codes(bits)

        codes = rearrange(codes, "... c d -> ... (c d)")

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)

        if project_out:
            codes = self.project_out(codes)

        # rearrange codes back to original shape

        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def forward(
        self,
        x: Tensor,
        inv_temperature=100.0,
        return_loss_breakdown=False,
        mask=None,
    ) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = x.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            x = rearrange(x, "b d ... -> b ... d")
            x, ps = pack_one(x, "b * d")

        assert (
            x.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but received {x.shape[-1]}"

        x = self.project_in(x)

        # split out number of codebooks

        x = rearrange(x, "b n (c d) -> b n c d", c=self.num_codebooks)

        # quantize by eq 3.

        original_input = x

        codebook_value = torch.ones_like(x) * self.codebook_scale
        quantized = torch.where(x > 0, codebook_value, -codebook_value)

        # use straight-through gradients (optionally with custom activation fn) if training

        if self.training:
            x = self.activation(x)
            x = x + (quantized - x).detach()
        else:
            x = quantized

        # calculate indices

        indices = reduce(
            (x > 0).int() * self.mask.int(), "b n c d -> b n c", "sum"
        )

        # entropy aux loss

        if self.training:
            # the same as euclidean distance up to a constant
            distance = -2 * einsum(
                "... i d, j d -> ... i j", original_input, self.codebook
            )

            prob = (-distance * inv_temperature).softmax(dim=-1)

            per_sample_entropy = entropy(prob).mean()

            # account for mask

            if exists(mask):
                prob = prob[mask]

            # distribution over all available tokens in the batch

            avg_prob = reduce(prob, "... c d -> c d", "mean")
            codebook_entropy = entropy(avg_prob).mean()

            # 1. entropy will be nudged to be low for each code, to encourage the network to output confident predictions
            # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used within the batch

            entropy_aux_loss = (
                per_sample_entropy - self.diversity_gamma * codebook_entropy
            )
        else:
            # if not training, just return dummy 0
            entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero

        # commit loss

        if self.training:
            commit_loss = F.mse_loss(
                original_input, quantized.detach(), reduction="none"
            )

            if exists(mask):
                commit_loss = commit_loss[mask]

            commit_loss = commit_loss.mean()
        else:
            commit_loss = self.zero

        # merge back codebook dim

        x = rearrange(x, "b n c d -> b n (c d)")

        # project out to feature dimension if needed

        x = self.project_out(x)

        # reconstitute image or video dimensions

        if is_img_or_video:
            x = unpack_one(x, ps, "b * d")
            x = rearrange(x, "b ... d -> b d ...")

            indices = unpack_one(indices, ps, "b * c")

        # whether to remove single codebook dim

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        # complete aux loss

        aux_loss = (
            entropy_aux_loss * self.entropy_loss_weight
            + commit_loss * self.commitment_loss_weight
        )

        ret = Return(x, indices, aux_loss)

        if not return_loss_breakdown:
            return ret

        return ret, LossBreakdown(
            per_sample_entropy, codebook_entropy, commit_loss
        )
