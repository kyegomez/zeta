import logging
import math
import time
from typing import Any, Callable, Dict, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList

try:
    from fairseq.modules.moe import MOELayer

    has_fairseq = True
    Base = MOELayer
except ModuleNotFoundError:
    Base = Module
    has_fairseq = False

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe

    has_tutel, fused_cumsum_sub_one = True, tutel_moe.fast_cumsum_sub_one
except ModuleNotFoundError:
    has_tutel, fused_cumsum_sub_one = (
        False,
        lambda mask: torch.cumsum(mask, dim=0) - 1,
    )

logger = logging.getLogger(__name__)


# use a fixed temperature to compute balance loss
TEMPERATURE_FOR_L_UAX = 0.07

# maximum capacity of 1 expert as a fraction of number of tokens in the batch
# Note: setting this to 1.0 causes inference to significantly slow down
EVAL_CAPACITY_TOKEN_FRACTION = 0.25

# logging
SAMPLE_FRACTION = 0.2


def _find_my_group_index(grouped_ranks):
    my_rank = dist.get_rank()
    for i, group in enumerate(grouped_ranks):
        if my_rank in group:
            return i
    raise RuntimeError


def get_moe_group(moe_expert_count=None):
    if dist.is_initialized():
        if not hasattr(get_moe_group, "_moe_groups"):
            world_size = dist.get_world_size()

            if world_size <= moe_expert_count:
                assert moe_expert_count % world_size == 0
                moe_groups = [[i] for i in range(world_size)]

            else:
                assert world_size % moe_expert_count == 0
                ranks_per_group = world_size // moe_expert_count
                moe_groups = [
                    [i + j * moe_expert_count for j in range(ranks_per_group)]
                    for i in range(moe_expert_count)
                ]

            get_moe_group._moe_expert_count = moe_expert_count
            get_moe_group._moe_group_idx = moe_groups
            get_moe_group._moe_groups = [dist.new_group(g) for g in moe_groups]

        my_group_idx = _find_my_group_index(get_moe_group._moe_group_idx)
        return my_group_idx, get_moe_group._moe_groups[my_group_idx]


def get_all2all_group(moe_expert_count):
    if dist.is_initialized():
        if not hasattr(get_all2all_group, "_all2all_groups"):
            world_size = dist.get_world_size()

            # more experts than world size
            if world_size <= moe_expert_count:
                assert moe_expert_count % world_size == 0
                all2all_groups = [list(range(world_size))]

            # larger world than num experts
            else:
                assert world_size % moe_expert_count == 0
                ranks_per_group = world_size // moe_expert_count
                all2all_groups = [
                    [i * moe_expert_count + j for j in range(moe_expert_count)]
                    for i in range(ranks_per_group)
                ]

            get_all2all_group._all2all_group_idx = all2all_groups
            get_all2all_group._all2all_groups = [
                dist.new_group(g) for g in all2all_groups
            ]

        my_group_idx = _find_my_group_index(
            get_all2all_group._all2all_group_idx
        )
        return get_all2all_group._all2all_groups[my_group_idx]


def top1gating(
    logits: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    use_fp32=False,
    capacity_factor=1.0,
    eval_mode=False,
    moe_eval_capacity_token_fraction=EVAL_CAPACITY_TOKEN_FRACTION,
    use_xmoe=False,
    gate_obj=None,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Implements Top2Gating on logits."""
    metadata = {}
    if use_fp32:
        orig_dtype = logits.dtype
        logits = logits.float()

    gates = F.softmax(logits, dim=1)
    metadata["entropy_gating"] = entropy(probs=gates).mean().detach()

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    if moe_eval_capacity_token_fraction > 0.0 and eval_mode:
        capacity = math.ceil(moe_eval_capacity_token_fraction * num_tokens)
    else:
        # capacity = capacity_factor * S/E
        capacity = int(capacity_factor * math.ceil(num_tokens / num_experts))

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = one_hot(indices1_s, num_classes=num_experts, unsqueeze_indices=True)
    if input_mask is not None and input_mask.any():
        nonpadding = ~input_mask
        mask1 = mask1 * nonpadding.unsqueeze(-1).to(mask1.dtype)

    # for logging (percent of tokens routed to each expert)
    expert1_hist = (
        100
        * torch.histc(
            (indices1_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts
        )
        / num_tokens
    )
    metadata["unused_expert1_count"] = (expert1_hist == 0).sum()
    expert1_hist = (
        torch.sort(expert1_hist, dim=0, descending=True).values
        + torch.finfo(torch.float32).tiny
    )

    sample_count = max(math.ceil(num_experts * SAMPLE_FRACTION), 1)
    metadata["expert1_balance_top"] = expert1_hist[:sample_count].sum()
    metadata["expert1_balance_bottom"] = expert1_hist[-sample_count:].sum()

    gates1_s = (gates * mask1).sum(dim=1)

    # Compute locations in capacity buffer
    locations1 = fused_cumsum_sub_one(mask1)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.to(gates.dtype), dim=0)

    l_aux = torch.mean(me * ce)
    l_aux = l_aux * num_experts * num_experts

    if has_tutel:
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        return (
            l_aux,
            metadata,
            capacity,
            num_experts,
            [
                indices1_s,
            ],
            [
                locations1_s,
            ],
            [
                gates1_s,
            ],
        )

    # Remove locations outside capacity from mask
    mask1 = mask1 * torch.lt(locations1, capacity)
    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Calculate combine_weights and dispatch_mask
    gates1 = gates1_s.unsqueeze(-1) * mask1.to(
        gates1_s.dtype
    )  # einsum("s,se->se")
    # locations1_sc = num_tokens * capacity
    locations1_sc = one_hot(
        locations1_s, num_classes=capacity, unsqueeze_indices=True
    )
    combine1_sec = torch.bmm(
        # einsum("se,sc->sec")
        gates1.unsqueeze(-1),
        locations1_sc.to(gates1.dtype).unsqueeze(1),
    )
    dispatch_mask = combine1_sec.bool()
    if use_fp32:
        return l_aux, combine1_sec.to(orig_dtype), dispatch_mask, metadata
    else:
        return l_aux, combine1_sec, dispatch_mask, metadata


class Top1Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        use_fp32=False,
        input_noise_type=None,
        capacity_factor=1.0,
        moe_eval_capacity_token_fraction=EVAL_CAPACITY_TOKEN_FRACTION,
        use_xmoe=False,
    ) -> None:
        # TODO: merge this to top2gate.py
        #
        super().__init__()

        if not use_xmoe:
            self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        else:
            self.wg_reduction = torch.nn.Linear(model_dim, 16, bias=False)
            wg = torch.empty(num_experts, 16)
            torch.nn.init.orthogonal_(wg, gain=0.32)
            self.register_parameter("wg", torch.nn.Parameter(wg))

        self.use_xmoe = use_xmoe
        self.use_fp32 = use_fp32
        self.input_noise_type = input_noise_type
        self.capacity_factor = capacity_factor
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction

    def forward(self, input, mask=None):  # type: ignore
        if self.use_xmoe:
            input = self.wg_reduction(input)
            with torch.no_grad():
                wg_norm = self.wg.norm(p=2.0, dim=1, keepdim=True)
                self.wg.mul_(1.5 / wg_norm)
            logits = self._cosine(input, self.wg)
            logits = self._make_finite(logits)
        else:
            logits = self.wg(input)

        return top1gating(
            logits,
            mask,
            use_fp32=self.use_fp32,
            capacity_factor=self.capacity_factor,
            eval_mode=not self.training,
            moe_eval_capacity_token_fraction=self.moe_eval_capacity_token_fraction,
            use_xmoe=self.use_xmoe,
            gate_obj=self,
        )

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores

    def _get_gating_temperature(self, eps=1e-4):
        if self.gating_t.data.item() < eps:
            return eps
        return self.gating_t

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)


gumbel_map: Dict[torch.device, Callable] = {}


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


def one_hot(
    indices: torch.Tensor, num_classes: int, unsqueeze_indices=False
) -> Tensor:
    if unsqueeze_indices:
        indices = indices.unsqueeze(-1)
    assert (
        indices.shape[-1] == 1
    ), "last dimension of indices must be have size 1"
    output = torch.zeros(
        indices.shape[:-1] + (num_classes,),
        device=indices.device,
        dtype=indices.dtype,
    )
    output.scatter_(len(output.shape) - 1, indices, 1)
    return output


def entropy(probs):
    logits = torch.distributions.utils.probs_to_logits(probs)
    p_log_p = probs * logits
    return -p_log_p.sum(-1)


def top2gating(
    logits: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    use_fp32=False,
    second_expert_policy="sampling",
    normalize_gate_prob_before_dropping=False,
    eval_mode=False,
    moe_eval_capacity_token_fraction=0.25,
    batch_prioritized_routing=False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    metadata = {}
    if use_fp32:
        orig_dtype = logits.dtype
        logits = logits.float()
    gates = F.softmax(logits, dim=1)
    metadata["entropy_gating"] = entropy(probs=gates).mean().detach()
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    if moe_eval_capacity_token_fraction > 0.0 and eval_mode:
        capacity = math.ceil(moe_eval_capacity_token_fraction * num_tokens)
    else:
        # capacity = 2S/E
        capacity = 2 * math.ceil(num_tokens / num_experts)

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1, keepdim=True)
    mask1 = one_hot(indices1_s, num_experts)
    if second_expert_policy == "sampling":
        # Create a mask for 2nd's expert per token using Gumbel-max trick
        # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
        logits_w_noise = logits + gumbel_rsample(
            logits.shape, device=logits.device
        )
    else:
        logits_w_noise = logits
    # Replace top-expert with min value
    logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1, keepdim=True)
    mask2 = one_hot(indices2_s, num_experts)
    gates1_s = (gates * mask1).sum(dim=1)
    gates2_s = (gates * mask2).sum(dim=1)

    if normalize_gate_prob_before_dropping:
        # Normalize gate probabilities
        denom_s = gates1_s + gates2_s
        # Avoid divide-by-zero
        denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
        gates1_s = gates1_s / denom_s
        gates2_s = gates2_s / denom_s

    if second_expert_policy == "random":
        sampled = (2 * gates2_s) > torch.rand_like(gates2_s)
        mask2 = mask2 * sampled.repeat(num_experts, 1).transpose(1, 0)

    # Compute locations in capacity buffer
    if input_mask is not None and input_mask.any():
        nonpadding = ~input_mask
        mask1 = mask1 * nonpadding.unsqueeze(-1).to(mask1.dtype)
        mask2 = mask2 * nonpadding.unsqueeze(-1).to(mask1.dtype)

    if batch_prioritized_routing:
        # if batch_prioritized_routing:
        importance_scores = -1 * gates.max(dim=1)[0]
        sorted_mask1 = mask1[importance_scores.argsort(dim=0)]
        sorted_cumsum1 = fused_cumsum_sub_one(sorted_mask1) * sorted_mask1
        importance_sorted_locations1 = sorted_cumsum1[
            importance_scores.argsort(dim=0).argsort(dim=0)
        ]

        sorted_mask2 = mask2[importance_scores.argsort(dim=0)]
        sorted_cumsum2 = fused_cumsum_sub_one(sorted_mask2) * sorted_mask2
        importance_sorted_locations2 = sorted_cumsum2[
            importance_scores.argsort(dim=0).argsort(dim=0)
        ]

        importance_sorted_locations2 += torch.sum(mask1, dim=0, keepdim=True)

        locations1, locations2 = (
            importance_sorted_locations1,
            importance_sorted_locations2,
        )
    else:
        locations1 = fused_cumsum_sub_one(mask1)
        locations2 = fused_cumsum_sub_one(mask2)
        # Update 2nd's location by accounting for locations of 1st
        locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.to(gates.dtype), dim=0)
    l_aux = torch.mean(me * ce)
    l_aux = l_aux * num_experts * num_experts

    # for logging purposes
    metadata["overflow_expert1"] = (
        100
        * torch.sum(mask1 * torch.ge(locations1, capacity))
        / torch.sum(mask1)
    )
    metadata["overflow_expert2"] = (
        100
        * torch.sum(mask2 * torch.ge(locations2, capacity))
        / torch.sum(mask2)
    )

    # Remove locations outside capacity from mask
    mask1_, mask2_ = mask1, mask2
    mask1 = mask1 * torch.lt(locations1, capacity)
    mask2 = mask2 * torch.lt(locations2, capacity)

    # for logging (percent of tokens routed to each expert)
    expert1_hist = (
        100
        * torch.histc(
            (indices1_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts
        )
        / num_tokens
    )
    metadata["unused_expert1_count"] = (expert1_hist == 0).sum()
    expert1_hist = (
        torch.sort(expert1_hist, dim=0, descending=True).values
        + torch.finfo(torch.float32).tiny
    )

    expert2_hist = (
        100
        * torch.histc(
            (indices2_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts
        )
        / num_tokens
    )
    metadata["unused_expert2_count"] = (expert2_hist == 0).sum()
    expert2_hist = (
        torch.sort(expert2_hist, dim=0, descending=True).values
        + torch.finfo(torch.float32).tiny
    )

    sample_count = max(math.ceil(num_experts * SAMPLE_FRACTION), 1)
    metadata["expert1_balance_top"] = expert1_hist[:sample_count].sum()
    metadata["expert1_balance_bottom"] = expert1_hist[-sample_count:].sum()

    metadata["expert2_balance_top"] = expert2_hist[:sample_count].sum()
    metadata["expert2_balance_bottom"] = expert2_hist[-sample_count:].sum()

    if not normalize_gate_prob_before_dropping:
        # Normalize gate probabilities
        gates1_s = (gates * mask1).sum(dim=1)
        gates2_s = (gates * mask2).sum(dim=1)
        denom_s = gates1_s + gates2_s
        # Avoid divide-by-zero
        denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
        gates1_s /= denom_s
        gates2_s /= denom_s

    if has_tutel:
        locations1_s = torch.sum(locations1 * mask1_, dim=1)
        locations2_s = torch.sum(locations2 * mask2_, dim=1)
        return (
            l_aux,
            metadata,
            capacity,
            num_experts,
            [indices1_s, indices2_s],
            [locations1_s, locations2_s],
            [gates1_s, gates2_s],
        )

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Calculate combine_weights and dispatch_mask
    gates1 = gates1_s.unsqueeze(-1) * mask1.to(
        gates1_s.dtype
    )  # einsum("s,se->se")
    gates2 = gates2_s.unsqueeze(-1) * mask2.to(
        gates2_s.dtype
    )  # einsum("s,se->se")
    locations1_sc = one_hot(
        locations1_s, num_classes=capacity, unsqueeze_indices=True
    )
    locations2_sc = one_hot(
        locations2_s, num_classes=capacity, unsqueeze_indices=True
    )
    combine1_sec = torch.bmm(
        # einsum("se,sc->sec")
        gates1.unsqueeze(-1),
        locations1_sc.to(gates1.dtype).unsqueeze(1),
    )
    combine2_sec = torch.bmm(
        # einsum("se,sc->sec")
        gates2.unsqueeze(-1),
        locations2_sc.to(gates2.dtype).unsqueeze(1),
    )
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()
    if use_fp32:
        return l_aux, combine_weights.to(orig_dtype), dispatch_mask, metadata
    else:
        return l_aux, combine_weights, dispatch_mask, metadata


class Top2Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        use_fp32=False,
        second_expert_policy="sampling",
        normalize_gate_prob_before_dropping=False,
        moe_eval_capacity_token_fraction=0.25,
        batch_prioritized_routing=False,
        use_xmoe=False,
    ) -> None:
        super().__init__()
        if not use_xmoe:
            self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        else:
            self.wg_reduction = torch.nn.Linear(model_dim, 16, bias=False)
            wg = torch.empty(num_experts, 16)
            torch.nn.init.orthogonal_(wg, gain=0.32)
            self.register_parameter("wg", torch.nn.Parameter(wg))
        self.use_fp32 = use_fp32
        self.second_expert_policy = second_expert_policy
        self.normalize_gate_prob_before_dropping = (
            normalize_gate_prob_before_dropping
        )
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction
        self.batch_prioritized_routing = batch_prioritized_routing
        self.use_xmoe = use_xmoe

    def forward(self, input, mask=None):  # type: ignore
        if self.use_xmoe:
            input = self.wg_reduction(input)
            with torch.no_grad():
                wg_norm = self.wg.norm(p=2.0, dim=1, keepdim=True)
                self.wg.mul_(1.5 / wg_norm)
            logits = self._cosine(input, self.wg)
            logits = self._make_finite(logits)
        else:
            logits = self.wg(input)
        return top2gating(
            logits,
            mask,
            use_fp32=self.use_fp32,
            second_expert_policy=self.second_expert_policy,
            normalize_gate_prob_before_dropping=self.normalize_gate_prob_before_dropping,
            eval_mode=not self.training,
            moe_eval_capacity_token_fraction=self.moe_eval_capacity_token_fraction,
            batch_prioritized_routing=self.batch_prioritized_routing,
        )

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        if torch.distributed.is_initialized():
            dist.all_to_all_single(output, input, group=group)
        else:
            assert group is None
            output = input
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


class GShardMoELayer(Base):
    """
    Mixture of Experts (MOE) layer implementation.

    Args:
        gate (nn.Module): The gating network that determines the expert assignment.
        experts (Union[nn.ModuleList, nn.Module]): The expert networks.
        args (argparse.Namespace): The command-line arguments.

    Attributes:
        gate (nn.Module): The gating network that determines the expert assignment.
        experts (nn.ModuleList): The expert networks.
        expert_group (dist.ProcessGroup): The process group for experts.
        all2all_group (dist.ProcessGroup): The process group for all-to-all communication.
        world_size (int): The number of processes in the expert group.
        all2all_size (int): The number of processes in the all-to-all group.
        num_local_experts (int): The number of local experts.
        args (argparse.Namespace): The command-line arguments.
        in_generation (bool): Flag indicating if the layer is in generation mode.
        a2a_cuda_event_intervals (List[Tuple[torch.cuda.Event, torch.cuda.Event]]): List of CUDA event intervals for all-to-all communication.
        a2a_cpu_time_ms (float): Total CPU time spent on all-to-all communication.

    Methods:
        forward(*input: Tensor, input_padding_mask=None, **kwargs: Any) -> Tensor:
            Performs forward pass through the MOE layer.
        prepare_for_inference_():
            Prepares the MOE layer for inference mode.
        all_to_all_wrapper(input: Tensor) -> Tensor:
            Wrapper function for all-to-all communication.
        record_all_to_all_stats():
            Records statistics for all-to-all communication.
    """

    def __init__(self, gate, experts, args):
        if has_fairseq:
            super(Base, self).__init__()
        else:
            super().__init__()
        self.gate = gate
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])
        _, self.expert_group = get_moe_group(args.moe_expert_count)
        self.all2all_group = get_all2all_group(args.moe_expert_count)
        self.world_size = dist.get_world_size(group=self.expert_group)
        self.all2all_size = dist.get_world_size(group=self.all2all_group)
        for p in experts.parameters():
            p.expert = True  # type: ignore
        self.num_local_experts = len(self.experts)
        self.args = args
        self.in_generation = False
        self.a2a_cuda_event_intervals = []
        self.a2a_cpu_time_ms = 0.0

    def forward(
        self, *input: Tensor, input_padding_mask=None, **kwargs: Any
    ) -> Tensor:
        assert len(input) == 1, "only single input Tensor supported"
        input = input[0]
        assert (
            len(input.shape) == 3
        ), "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        if input_padding_mask is not None:
            assert (
                len(input_padding_mask.shape) == 2
            ), "input Tensor must have dimensions: (s)equence, (t)oken"
            assert input_padding_mask.shape[0] == input.shape[0]
            assert input_padding_mask.shape[1] == input.shape[1]
        # assert input.shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"

        # Implement Algorithm 2 from GShard paper.
        dim = input.shape[2]
        # Pad to expected batch size
        input_shape = list(input.shape)
        expected_bsz = (
            getattr(self.args, "batch_size", 0)
            if self.training
            else getattr(self.args, "batch_size_valid", 0)
        )
        # This indicates that --batch-size or --max-sentences is not specified
        if expected_bsz is None:
            expected_bsz = 0
        # Note: Padding is not necessary at generation time at present
        # because all DDP workers process the same batch. Also, batch size at generation time
        # can be different from that present in the checkpoint state
        if (
            not self.in_generation
            and expected_bsz != 0
            and input_shape[0] != expected_bsz
        ):
            logger.warning(
                "padding batch with unexpected size"
                f" {input_shape[0]} (expected: {expected_bsz})"
            )
            assert (
                input_shape[0] < expected_bsz
            ), f"{input_shape[0]} < {expected_bsz}"
            padded_input = torch.zeros(
                (expected_bsz, input_shape[1], input_shape[2]),
                dtype=input.dtype,
                layout=input.layout,
                device=input.device,
            )
            padded_input[: input_shape[0], :, :] = input
            input = padded_input

            padded_input_padding_mask = torch.ones(
                (
                    expected_bsz,
                    input_shape[1],
                ),
                dtype=torch.bool,
                device=input.device,
            )
            if input_padding_mask is not None:
                padded_input_padding_mask[: input_shape[0], :] = (
                    input_padding_mask
                )
            else:
                padded_input_padding_mask[: input_shape[0], :] = False
            input_padding_mask = padded_input_padding_mask

        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input.reshape(-1, dim)
        reshaped_input_shape = reshaped_input.shape
        reshaped_input_padding_mask = (
            input_padding_mask.reshape(-1)
            if input_padding_mask is not None
            else None
        )

        # Doing padding here when --max-tokens is specified and not --batch-size or --max-sentences
        # Pro of --max-tokens: more flexible for MT variable sequence lengths
        # Con of --max-tokens: extra all-reduce needed to figure out optimal padding without running OOM
        if expected_bsz == 0:
            expected_dim = reshaped_input_shape[0] * torch.ones(
                (1,), dtype=torch.long, device=input.device
            )
            dist.all_reduce(
                expected_dim, group=dist.group.WORLD, op=dist.ReduceOp.MAX
            )
            expected_dim = int(expected_dim.item())
            padded_input = torch.zeros(
                (expected_dim, reshaped_input_shape[1]),
                dtype=input.dtype,
                layout=input.layout,
                device=input.device,
            )
            padded_input[: reshaped_input_shape[0], :] = reshaped_input
            reshaped_input = padded_input

            padded_input_padding_mask = torch.ones(
                (expected_dim,), dtype=torch.bool, device=padded_input.device
            )
            if reshaped_input_padding_mask is not None:
                padded_input_padding_mask[: reshaped_input_shape[0]] = (
                    reshaped_input_padding_mask
                )
            else:
                padded_input_padding_mask[: reshaped_input_shape[0]] = False
            reshaped_input_padding_mask = padded_input_padding_mask

        if has_tutel:
            (
                l_aux,
                self.metadata,
                C,
                E,
                indices_,
                locations_,
                gates_,
            ) = self.gate(reshaped_input, reshaped_input_padding_mask)
            S, M = reshaped_input.size(0), reshaped_input.size(1)

            if not hasattr(self, "_tutel_dispatcher"):
                self._tutel_dispatcher = tutel_moe.fast_dispatcher(
                    E, C, M, dispatch_dtype=reshaped_input.dtype
                )
            self._tutel_dispatcher.update(
                indices_, locations_, gates_, capacity=C
            )
            dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
        else:
            l_aux, combine_weights, dispatch_mask, self.metadata = self.gate(
                reshaped_input, reshaped_input_padding_mask
            )

            dispatch_mask = dispatch_mask.to(input.dtype).permute(
                1, 2, 0
            )  # S,E,C -> E,C,S
            E, C, S = dispatch_mask.size()
            M = reshaped_input.size(1)
            assert reshaped_input.size() == (S, M)
            # einsum("sec,sm->ecm")
            dispatched_input = torch.mm(
                dispatch_mask.view(E * C, S), reshaped_input
            )  # -> (E*C),M

        if self.all2all_size > 1:
            dispatched_input = self.all_to_all_wrapper(dispatched_input)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(
            self.all2all_size, self.num_local_experts, -1, dim
        )
        chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=1)

        if self.all2all_size > 1:
            expert_output = self.all_to_all_wrapper(expert_output)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(
            self.all2all_size * self.num_local_experts, -1, dim
        )

        if has_tutel:
            combined_output = self._tutel_dispatcher.decode(
                expert_output.view(E * C, M)
            )
        else:
            # einsum("sec,ecm->sm")
            combined_output = combine_weights.view(S, E * C).mm(
                expert_output.view(E * C, M)
            )

        # Remove padding here when --max-tokens is specified and not --batch-size or --max-sentences
        combined_output = combined_output[: reshaped_input_shape[0], :]
        combined_output = combined_output.reshape(input.shape)
        combined_output = combined_output[: input_shape[0], :, :]

        self.record_all_to_all_stats()

        return combined_output, l_aux

    def prepare_for_inference_(self):
        self.in_generation = True

    def all_to_all_wrapper(self, input: Tensor):
        dummy_a2a = getattr(self.args, "dummy_a2a", False)
        if dummy_a2a:
            input = input.contiguous()
            output = input.detach().clone()
            return input
        # always record times, since it is not a lot of overhead
        # if we do not log it we simply clear it off in record_all_to_all_stats
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cpu_start = time.time() * 1000
        cuda_start.record()
        output = _AllToAll.apply(self.all2all_group, input)
        cuda_end.record()
        cpu_end = time.time() * 1000
        self.a2a_cpu_time_ms += cpu_end - cpu_start
        self.a2a_cuda_event_intervals.append((cuda_start, cuda_end))
        return output

    def record_all_to_all_stats(self):
        # controlled via an argument as we want to minimize any impact from torch.cuda.synchronize()
        record_a2a_perf_stats = getattr(
            self.args, "record_a2a_perf_stats", False
        )
        if record_a2a_perf_stats:
            torch.cuda.synchronize()
            self.metadata["all_to_all_cpu_time_ms"] = self.a2a_cpu_time_ms
            a2a_cuda_time_ms = 0.0
            for ev_start, ev_end in self.a2a_cuda_event_intervals:
                a2a_cuda_time_ms += ev_start.elapsed_time(ev_end)
            self.metadata["all_to_all_cuda_time_ms"] = a2a_cuda_time_ms
        # reset stats
        self.a2a_cpu_time_ms = 0.0
        self.a2a_cuda_event_intervals = []
