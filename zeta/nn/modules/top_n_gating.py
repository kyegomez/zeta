from functools import partial
from typing import Tuple, Union

import torch
from torch.nn import Module
from torch import nn
import torch.nn.functional as F

from beartype import beartype

from einops import rearrange, reduce

from colt5_attention import topk as maybe_differentiable_topk


def cast_tuple(el, len=1):
    return el if isinstance(el, tuple) else ((el,) * len)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def cumsum_exclusive(t, dim=-3):
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim=dim)


def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    one_hot_classes = max(max_index + 1, max_length)
    return F.one_hot(indexes, one_hot_classes)[..., :max_length]


class TopNGating(Module):
    """TopNGating

    Args:
        dim (int): The input dimension.
        num_gates (int): The number of gates.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-9.
        top_n (int, optional): The number of experts to route to. Defaults to 2.
        threshold_train (Union[float, Tuple[float, ...]], optional): The threshold for routing to the top-n experts during training. Defaults to 0.2.
        threshold_eval (Union[float, Tuple[float, ...]], optional): The threshold for routing to the top-n experts during evaluation. Defaults to 0.2.
        capacity_factor_train (float, optional): The capacity factor for routing to the top-n experts during training. Defaults to 1.25.
        capacity_factor_eval (float, optional): The capacity factor for routing to the top-n experts during evaluation. Defaults to 2.0.
        straight_through_dispatch_tensor (bool, optional): Whether to use the straight-through version of the dispatch tensor. Defaults to True.
        differentiable_topk (bool, optional): Whether to use the differentiable version of the top-k operation. Defaults to False.
        differentiable_topk_fused (bool, optional): Whether to use the fused version of the differentiable top-k operation. Defaults to True.
        min_expert_capacity (int, optional): The minimum capacity of each expert. Defaults to 4.

    Examples:
        x = torch.randn(1, 2, 3)
        model = TopNGating(3, 4)
        out, _, _, _, = model(x)
        print(out.shape)


    """

    @beartype
    def __init__(
        self,
        dim,
        num_gates,
        eps=1e-9,
        top_n=2,
        threshold_train: Union[float, Tuple[float, ...]] = 0.2,
        threshold_eval: Union[float, Tuple[float, ...]] = 0.2,
        capacity_factor_train=1.25,
        capacity_factor_eval=2.0,
        straight_through_dispatch_tensor=True,
        differentiable_topk=False,
        differentiable_topk_fused=True,
        min_expert_capacity: int = 4,
    ):
        super().__init__()
        self.eps = eps
        self.num_gates = num_gates
        self.min_expert_capacity = min_expert_capacity
        self.to_gates = nn.Linear(dim, num_gates, bias=False)

        self.differentiable_topk = differentiable_topk

        self.topk = partial(
            maybe_differentiable_topk,
            non_differentiable=not differentiable_topk,
            fused=differentiable_topk_fused,  # use triton fused coordinate descent if possible by default
        )

        assert top_n >= 2, "must be 2 or more experts"
        self.top_n = top_n
        top_n_minus_1 = top_n - 1

        threshold_train = cast_tuple(threshold_train, top_n_minus_1)
        threshold_eval = cast_tuple(threshold_eval, top_n_minus_1)

        assert len(threshold_train) == len(threshold_eval) == top_n_minus_1

        self.register_buffer(
            "threshold_train", torch.tensor([eps, *threshold_train])
        )
        self.register_buffer(
            "threshold_eval", torch.tensor([eps, *threshold_eval])
        )

        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

        self.straight_through_dispatch_tensor = straight_through_dispatch_tensor
        self.register_buffer("zero", torch.zeros((1,)), persistent=False)

    def forward(self, x, noise_gates=False, noise_mult=1.0):
        """
        einstein notation:

        b - batch
        n - sequence
        e - experts
        k - top-n experts
        """

        *_, b, group_size, dim, dtype, top_n, num_gates, eps = (
            *x.shape,
            x.dtype,
            self.top_n,
            self.num_gates,
            self.eps,
        )

        # threshold, capacity depending on training or eval

        suffix = "train" if self.training else "eval"

        threshold = getattr(self, f"threshold_{suffix}")
        capacity_factor = getattr(self, f"capacity_factor_{suffix}")

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes

        expert_capacity = min(
            group_size, int((group_size * capacity_factor) / num_gates)
        )
        expert_capacity = max(expert_capacity, self.min_expert_capacity)
        expert_capacity_f = float(expert_capacity)

        # gate logits and gates

        gate_logits = self.to_gates(x)

        maybe_noised_gate_logits = gate_logits

        if noise_gates:
            noise = gumbel_noise(maybe_noised_gate_logits)
            maybe_noised_gate_logits = (
                maybe_noised_gate_logits + noise * noise_mult
            )

        raw_gates = maybe_noised_gate_logits.softmax(dim=-1)

        # find top N experts per position

        topk_return = self.topk(raw_gates, k=top_n)

        gate_indices = topk_return.indices

        if self.differentiable_topk:
            # allow for differentiable topk using coordinate descent
            # used successfully for routing from CoLT5 paper https://github.com/lucidrains/CoLT5-attention

            gates = topk_return.coor_descent_values
        else:
            gates = topk_return.values

        # move the top-n dimension to be first

        gates = rearrange(gates, "... k -> k ...")
        gate_indices = rearrange(gate_indices, "... k -> k ...")

        # masks

        one_hot_gate_indices = F.one_hot(gate_indices, num_gates)
        mask = one_hot_gate_indices.float()

        mask_1 = mask[0]  # needed for balancing loss

        # normalize top-n gate scores

        denom = reduce(gates, "k ... -> 1 ...", "sum").clamp(min=eps)
        gates = gates / denom

        # best performing policy was to route to the second expert, with probability of min(1., score / threshold), where score = gate2 / (gate1 + gate2)
        # optimal threshold was ~ 0.2
        # generalized to more than 2 experts

        probs = torch.zeros_like(gates).uniform_(0.0, 1.0)

        threshold = rearrange(threshold, "k -> k 1 1")
        should_route = probs < (gates / threshold.clamp(min=eps))

        # tokens should always be routed to first expert
        # threshold for first expert already set to very small number, but just in case

        should_route[0, ...] = True

        mask *= rearrange(should_route.float(), "... -> ... 1")

        mask_cumsum = cumsum_exclusive(mask, dim=-2)  # along sequence dimension

        # compute assignment to experts - (batch, seq, experts)

        # This is the position within the expert's mini-batch for this sequence

        positions = []
        prev_expert_count = 0.0

        for n in range(self.top_n):
            position_in_expert = (mask_cumsum[n] + prev_expert_count) * mask[n]

            # Remove the elements that don't fit. (batch, sequence, experts)
            mask[n] *= (position_in_expert < expert_capacity_f).float()

            # How many examples in this sequence go to this expert - needed for the next iteration as offset
            prev_expert_count = reduce(mask[n], "... n e -> ... 1 e", "sum")

            # (batch, sequence)
            position_in_expert = reduce(
                position_in_expert, "... n e -> ... n", "sum"
            )
            positions.append(position_in_expert)

        positions = torch.stack(positions)

        # (k, batch, sequence) - mostly ones, but zeros where something didn't fit
        mask_flat = reduce(mask, "... n e -> ... n", "sum")

        # (k, batch, sequence) - weighted assignment
        # following https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py#L1903
        gates = gates * mask_flat

        # (batch, sequence, experts, expert_capacity)

        N = None

        gates = gates[..., N, N]
        mask_flat = mask_flat[..., N, N]
        one_hot_gate_indices = one_hot_gate_indices[..., N]
        safe_one_hot_gates = safe_one_hot(positions.long(), expert_capacity)[
            ..., N, :
        ]

        combine_tensor = reduce(
            gates * mask_flat * one_hot_gate_indices * safe_one_hot_gates,
            "k ... -> ...",
            "sum",
        )

        # dispatch tensor

        dispatch_tensor = combine_tensor.bool().type(dtype)

        if self.straight_through_dispatch_tensor:
            dispatch_tensor = (
                dispatch_tensor + combine_tensor - combine_tensor.detach()
            )

        # balance losses - (batch, experts)
        # We want to equalize the fraction of the batch assigned to each expert

        if self.training:
            density_1 = reduce(mask_1, "... n e -> ... e", "mean")
            density_1_proxy = reduce(
                raw_gates, "... n e -> ... e", "mean"
            )  # Something continuous that is correlated with what we want to equalize.

            balance_loss = (density_1_proxy * density_1).mean() * float(
                num_gates**2
            )
        else:
            balance_loss = self.zero

        # calculate the router z-loss proposed in paper

        if self.training:
            router_z_loss = torch.logsumexp(gate_logits, dim=-1)
            router_z_loss = torch.square(router_z_loss)
            router_z_loss = router_z_loss.mean()
        else:
            router_z_loss = self.zero

        return dispatch_tensor, combine_tensor, balance_loss, router_z_loss
