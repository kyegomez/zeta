import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


class MOELayer(nn.Module):
    """
    Base Mixture of Experts (MoE) layer implementation using pure PyTorch.

    This class serves as a base class for MoE implementations, providing
    a common interface and basic functionality. It can be extended to implement
    specific MoE architectures.

    Args:
        gate (nn.Module): The gating network that determines expert assignment.
        experts (Union[nn.ModuleList, nn.Module]): The expert networks.
        args (object): Configuration object containing MoE parameters.

    Attributes:
        gate (nn.Module): The gating network.
        experts (nn.ModuleList): List of expert networks.
        metadata (Dict[str, Any]): Dictionary to store routing statistics and metrics.
        l_aux (Optional[Tensor]): Auxiliary loss for load balancing.
    """

    def __init__(
        self,
        gate: nn.Module,
        experts: Union[nn.ModuleList, nn.Module],
        args: Any,
    ) -> None:
        super().__init__()
        self.gate = gate
        if isinstance(experts, nn.ModuleList):
            self.experts = experts
        else:
            self.experts = nn.ModuleList([experts])

        self.args = args
        self.metadata: Dict[str, Any] = {}
        self.l_aux: Optional[Tensor] = None

        # Mark expert parameters for distributed training
        for expert in self.experts:
            for param in expert.parameters():
                param.expert = True  # type: ignore

    def forward(
        self,
        input_tensor: Tensor,
        input_padding_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass through the MoE layer."""
        raise NotImplementedError(
            "MOELayer is a base class. Use a specific implementation like GShardMoELayer."
        )

    def prepare_for_inference_(self) -> None:
        """Prepare the layer for inference mode."""
        pass

    def get_aux_loss(self) -> Optional[Tensor]:
        """Get the auxiliary loss for load balancing."""
        return self.l_aux

    def get_metadata(self) -> Dict[str, Any]:
        """Get routing metadata and statistics."""
        return self.metadata.copy()

    def reset_metadata(self) -> None:
        """Reset the metadata dictionary."""
        self.metadata.clear()
        self.l_aux = None


class FastDispatcher:
    """
    Custom implementation for efficient token dispatching in MoE layers.

    This class provides efficient dispatching of tokens to experts and
    combines expert outputs back to the original token order.
    """

    def __init__(
        self,
        num_experts: int,
        capacity: int,
        model_dim: int,
        dispatch_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_experts = num_experts
        self.capacity = capacity
        self.model_dim = model_dim
        self.dispatch_dtype = dispatch_dtype

        # Initialize state
        self.indices: Optional[Tensor] = None
        self.locations: Optional[Tensor] = None
        self.gates: Optional[Tensor] = None

    def update(
        self,
        indices: Union[Tensor, List[Tensor]],
        locations: Union[Tensor, List[Tensor]],
        gates: Union[Tensor, List[Tensor]],
        capacity: Optional[int] = None,
    ) -> None:
        """Update dispatcher state with routing information."""
        if capacity is not None:
            self.capacity = capacity

        # Handle both tensor and list inputs
        if isinstance(indices, list):
            self.indices = indices[0] if len(indices) > 0 else None
        else:
            self.indices = indices

        if isinstance(locations, list):
            self.locations = locations[0] if len(locations) > 0 else None
        else:
            self.locations = locations

        if isinstance(gates, list):
            self.gates = gates[0] if len(gates) > 0 else None
        else:
            self.gates = gates

    def encode(self, input_tensor: Tensor) -> Tensor:
        """Dispatch tokens to experts based on routing decisions."""
        if self.indices is None or self.locations is None or self.gates is None:
            raise RuntimeError("Must call update() before encode()")

        num_tokens, model_dim = input_tensor.shape
        device = input_tensor.device

        # Create dispatch mask: (num_tokens, num_experts * capacity)
        dispatch_mask = torch.zeros(
            (num_tokens, self.num_experts * self.capacity),
            dtype=self.dispatch_dtype,
            device=device,
        )

        valid_locations = self.locations < self.capacity
        for i in range(num_tokens):
            if valid_locations[i]:
                expert_idx = self.indices[i].item()
                location_idx = self.locations[i].item()
                flat_idx = expert_idx * self.capacity + location_idx
                dispatch_mask[i, flat_idx] = self.gates[i].item()

        # Dispatch: (num_experts * capacity, model_dim)
        dispatched = dispatch_mask.t() @ input_tensor.to(self.dispatch_dtype)

        return dispatched

    def decode(self, expert_output: Tensor) -> Tensor:
        """Combine expert outputs back to original token order."""
        if self.indices is None or self.locations is None or self.gates is None:
            raise RuntimeError("Must call update() before decode()")

        device = expert_output.device
        num_tokens = self.indices.shape[0]

        # Create combine mask: (num_tokens, num_experts * capacity)
        combine_mask = torch.zeros(
            (num_tokens, self.num_experts * self.capacity),
            dtype=self.dispatch_dtype,
            device=device,
        )

        valid_locations = self.locations < self.capacity
        for i in range(num_tokens):
            if valid_locations[i]:
                expert_idx = self.indices[i].item()
                location_idx = self.locations[i].item()
                flat_idx = expert_idx * self.capacity + location_idx
                combine_mask[i, flat_idx] = self.gates[i].item()

        # Combine: (num_tokens, model_dim)
        combined = combine_mask @ expert_output

        return combined


def fast_cumsum_sub_one(mask: Tensor) -> Tensor:
    """Compute cumulative sum along dimension 0 minus 1."""
    return torch.cumsum(mask, dim=0) - 1


# Use a fixed temperature to compute balance loss
TEMPERATURE_FOR_L_UAX = 0.07

# Maximum capacity of 1 expert as a fraction of number of tokens in the batch
EVAL_CAPACITY_TOKEN_FRACTION = 0.25

# Logging
SAMPLE_FRACTION = 0.2


def _find_my_group_index(grouped_ranks: List[List[int]]) -> int:
    """Find the index of the group containing the current process rank."""
    my_rank = dist.get_rank()
    for i, group in enumerate(grouped_ranks):
        if my_rank in group:
            return i
    raise RuntimeError(f"Rank {my_rank} not found in any group")


def get_moe_group(moe_expert_count: Optional[int] = None) -> Tuple[int, Any]:
    """Get the MoE process group for expert parallelism."""
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
    return 0, None


def get_all2all_group(moe_expert_count: int) -> Any:
    """Get the all-to-all process group for MoE communication."""
    if dist.is_initialized():
        if not hasattr(get_all2all_group, "_all2all_groups"):
            world_size = dist.get_world_size()

            # More experts than world size
            if world_size <= moe_expert_count:
                assert moe_expert_count % world_size == 0
                all2all_groups = [list(range(world_size))]
            # Larger world than num experts
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
    return None


def one_hot(
    indices: Tensor, num_classes: int, unsqueeze_indices: bool = False
) -> Tensor:
    """Create one-hot encoding of indices."""
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


def entropy(probs: Tensor) -> Tensor:
    """Compute entropy of probability distributions."""
    logits = torch.distributions.utils.probs_to_logits(probs)
    p_log_p = probs * logits
    return -p_log_p.sum(-1)


def gumbel_rsample(shape: Tuple[int, ...], device: torch.device) -> Tensor:
    """Sample from Gumbel distribution."""
    uniform = torch.rand(shape, device=device)
    return -torch.log(-torch.log(uniform + 1e-8) + 1e-8)


def top1gating(
    logits: Tensor,
    input_mask: Optional[Tensor] = None,
    use_fp32: bool = False,
    capacity_factor: float = 1.0,
    eval_mode: bool = False,
    moe_eval_capacity_token_fraction: float = EVAL_CAPACITY_TOKEN_FRACTION,
    use_xmoe: bool = False,
    gate_obj: Any = None,
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
    """Implements Top1 gating for MoE."""
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

    # Create a mask for 1st expert per token
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = one_hot(indices1_s, num_classes=num_experts, unsqueeze_indices=True)
    if input_mask is not None and input_mask.any():
        nonpadding = ~input_mask
        mask1 = mask1 * nonpadding.unsqueeze(-1).to(mask1.dtype)

    # For logging (percent of tokens routed to each expert)
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
    locations1 = fast_cumsum_sub_one(mask1)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.to(gates.dtype), dim=0)

    l_aux = torch.mean(me * ce)
    l_aux = l_aux * num_experts * num_experts

    # Remove locations outside capacity from mask
    mask1 = mask1 * torch.lt(locations1, capacity)
    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Calculate combine_weights and dispatch_mask
    gates1 = gates1_s.unsqueeze(-1) * mask1.to(gates1_s.dtype)
    locations1_sc = one_hot(
        locations1_s, num_classes=capacity, unsqueeze_indices=True
    )
    combine1_sec = torch.bmm(
        gates1.unsqueeze(-1),
        locations1_sc.to(gates1.dtype).unsqueeze(1),
    )
    dispatch_mask = combine1_sec.bool()

    if use_fp32:
        return l_aux, combine1_sec.to(orig_dtype), dispatch_mask, metadata
    else:
        return l_aux, combine1_sec, dispatch_mask, metadata


def top2gating(
    logits: Tensor,
    input_mask: Optional[Tensor] = None,
    use_fp32: bool = False,
    second_expert_policy: str = "sampling",
    normalize_gate_prob_before_dropping: bool = False,
    eval_mode: bool = False,
    moe_eval_capacity_token_fraction: float = 0.25,
    batch_prioritized_routing: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
    """Implements Top2 gating for MoE."""
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

    # Create a mask for 1st expert per token
    indices1_s = torch.argmax(gates, dim=1, keepdim=True)
    mask1 = one_hot(indices1_s, num_experts)

    if second_expert_policy == "sampling":
        # Create a mask for 2nd expert per token using Gumbel-max trick
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
        importance_scores = -1 * gates.max(dim=1)[0]
        sorted_mask1 = mask1[importance_scores.argsort(dim=0)]
        sorted_cumsum1 = fast_cumsum_sub_one(sorted_mask1) * sorted_mask1
        importance_sorted_locations1 = sorted_cumsum1[
            importance_scores.argsort(dim=0).argsort(dim=0)
        ]

        sorted_mask2 = mask2[importance_scores.argsort(dim=0)]
        sorted_cumsum2 = fast_cumsum_sub_one(sorted_mask2) * sorted_mask2
        importance_sorted_locations2 = sorted_cumsum2[
            importance_scores.argsort(dim=0).argsort(dim=0)
        ]

        importance_sorted_locations2 += torch.sum(mask1, dim=0, keepdim=True)

        locations1, locations2 = (
            importance_sorted_locations1,
            importance_sorted_locations2,
        )
    else:
        locations1 = fast_cumsum_sub_one(mask1)
        locations2 = fast_cumsum_sub_one(mask2)
        # Update 2nd's location by accounting for locations of 1st
        locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.to(gates.dtype), dim=0)
    l_aux = torch.mean(me * ce)
    l_aux = l_aux * num_experts * num_experts

    # For logging purposes
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
    mask1 = mask1 * torch.lt(locations1, capacity)
    mask2 = mask2 * torch.lt(locations2, capacity)

    # For logging (percent of tokens routed to each expert)
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
        denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
        gates1_s /= denom_s
        gates2_s /= denom_s

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Calculate combine_weights and dispatch_mask
    gates1 = gates1_s.unsqueeze(-1) * mask1.to(gates1_s.dtype)
    gates2 = gates2_s.unsqueeze(-1) * mask2.to(gates2_s.dtype)
    locations1_sc = one_hot(
        locations1_s, num_classes=capacity, unsqueeze_indices=True
    )
    locations2_sc = one_hot(
        locations2_s, num_classes=capacity, unsqueeze_indices=True
    )
    combine1_sec = torch.bmm(
        gates1.unsqueeze(-1),
        locations1_sc.to(gates1.dtype).unsqueeze(1),
    )
    combine2_sec = torch.bmm(
        gates2.unsqueeze(-1),
        locations2_sc.to(gates2.dtype).unsqueeze(1),
    )
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    if use_fp32:
        return l_aux, combine_weights.to(orig_dtype), dispatch_mask, metadata
    else:
        return l_aux, combine_weights, dispatch_mask, metadata


class Top1Gate(nn.Module):
    """Top-1 gating mechanism for MoE."""

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        use_fp32: bool = False,
        input_noise_type: Optional[str] = None,
        capacity_factor: float = 1.0,
        moe_eval_capacity_token_fraction: float = EVAL_CAPACITY_TOKEN_FRACTION,
        use_xmoe: bool = False,
    ) -> None:
        super().__init__()

        if not use_xmoe:
            self.wg = nn.Linear(model_dim, num_experts, bias=False)
        else:
            self.wg_reduction = nn.Linear(model_dim, 16, bias=False)
            wg = torch.empty(num_experts, 16)
            nn.init.orthogonal_(wg, gain=0.32)
            self.register_parameter("wg", nn.Parameter(wg))

        self.use_xmoe = use_xmoe
        self.use_fp32 = use_fp32
        self.input_noise_type = input_noise_type
        self.capacity_factor = capacity_factor
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction

    def forward(
        self, input_tensor: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
        if self.use_xmoe:
            input_tensor = self.wg_reduction(input_tensor)
            with torch.no_grad():
                wg_norm = self.wg.norm(p=2.0, dim=1, keepdim=True)
                self.wg.mul_(1.5 / wg_norm)
            logits = self._cosine(input_tensor, self.wg)
            logits = self._make_finite(logits)
        else:
            logits = self.wg(input_tensor)

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

    def _make_finite(self, scores: Tensor) -> Tensor:
        ok = scores.isfinite()
        if not ok.all():
            scores[~ok] = scores[ok].min()
        return scores

    def _cosine(self, mat1: Tensor, mat2: Tensor, eps: float = 1e-4) -> Tensor:
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)


class Top2Gate(nn.Module):
    """Top-2 gating mechanism for MoE."""

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        use_fp32: bool = False,
        second_expert_policy: str = "sampling",
        normalize_gate_prob_before_dropping: bool = False,
        moe_eval_capacity_token_fraction: float = 0.25,
        batch_prioritized_routing: bool = False,
        use_xmoe: bool = False,
    ) -> None:
        super().__init__()
        if not use_xmoe:
            self.wg = nn.Linear(model_dim, num_experts, bias=False)
        else:
            self.wg_reduction = nn.Linear(model_dim, 16, bias=False)
            wg = torch.empty(num_experts, 16)
            nn.init.orthogonal_(wg, gain=0.32)
            self.register_parameter("wg", nn.Parameter(wg))

        self.use_fp32 = use_fp32
        self.second_expert_policy = second_expert_policy
        self.normalize_gate_prob_before_dropping = (
            normalize_gate_prob_before_dropping
        )
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction
        self.batch_prioritized_routing = batch_prioritized_routing
        self.use_xmoe = use_xmoe

    def forward(
        self, input_tensor: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
        if self.use_xmoe:
            input_tensor = self.wg_reduction(input_tensor)
            with torch.no_grad():
                wg_norm = self.wg.norm(p=2.0, dim=1, keepdim=True)
                self.wg.mul_(1.5 / wg_norm)
            logits = self._cosine(input_tensor, self.wg)
            logits = self._make_finite(logits)
        else:
            logits = self.wg(input_tensor)

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

    def _cosine(self, mat1: Tensor, mat2: Tensor, eps: float = 1e-4) -> Tensor:
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def _make_finite(self, scores: Tensor) -> Tensor:
        ok = scores.isfinite()
        if not ok.all():
            scores[~ok] = scores[ok].min()
        return scores


class _AllToAll(torch.autograd.Function):
    """All-to-all communication primitive."""

    @staticmethod
    def forward(ctx: Any, group: Any, input_tensor: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input_tensor = input_tensor.contiguous()
        output = torch.empty_like(input_tensor)
        if torch.distributed.is_initialized() and group is not None:
            dist.all_to_all_single(output, input_tensor, group=group)
        else:
            output = input_tensor
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


class GShardMoELayer(nn.Module):
    """
    GShard Mixture of Experts (MoE) layer implementation using pure PyTorch.

    This implementation follows the GShard paper architecture for distributed
    mixture of experts, supporting both Top-1 and Top-2 gating mechanisms
    with load balancing and expert parallelism.

    Args:
        gate (nn.Module): The gating network (Top1Gate or Top2Gate).
        experts (Union[nn.ModuleList, nn.Module]): The expert networks.
        args (object): Configuration object with MoE parameters.

    Example:
        >>> gate = Top2Gate(model_dim=512, num_experts=8)
        >>> expert = nn.Linear(512, 512)
        >>> class Args:
        ...     moe_expert_count = 8
        >>> args = Args()
        >>> moe = GShardMoELayer(gate, expert, args)
        >>> x = torch.randn(32, 128, 512)
        >>> output, aux_loss = moe(x)
    """

    def __init__(
        self,
        gate: nn.Module,
        experts: Union[nn.ModuleList, nn.Module],
        args: Any,
    ) -> None:
        super().__init__()
        self.gate = gate
        if isinstance(experts, nn.ModuleList):
            self.experts = experts
        else:
            self.experts = nn.ModuleList([experts])

        _, self.expert_group = get_moe_group(args.moe_expert_count)
        self.all2all_group = get_all2all_group(args.moe_expert_count)

        if dist.is_initialized():
            self.world_size = dist.get_world_size(group=self.expert_group)
            self.all2all_size = dist.get_world_size(group=self.all2all_group)
        else:
            self.world_size = 1
            self.all2all_size = 1

        for expert in self.experts:
            for p in expert.parameters():
                p.expert = True  # type: ignore

        self.num_local_experts = len(self.experts)
        self.args = args
        self.in_generation = False
        self.a2a_cuda_event_intervals: List[
            Tuple[torch.cuda.Event, torch.cuda.Event]
        ] = []
        self.a2a_cpu_time_ms = 0.0
        self.metadata: Dict[str, Any] = {}

    def forward(
        self,
        *input_args: Tensor,
        input_padding_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass through the GShard MoE layer."""
        assert len(input_args) == 1, "only single input Tensor supported"
        input_tensor = input_args[0]
        assert (
            len(input_tensor.shape) == 3
        ), "input Tensor must have dimensions: (batch, sequence, model)"

        if input_padding_mask is not None:
            assert len(input_padding_mask.shape) == 2
            assert input_padding_mask.shape[0] == input_tensor.shape[0]
            assert input_padding_mask.shape[1] == input_tensor.shape[1]

        # Implement Algorithm 2 from GShard paper
        dim = input_tensor.shape[2]
        input_shape = list(input_tensor.shape)

        # Handle batch size padding
        expected_bsz = (
            getattr(self.args, "batch_size", 0)
            if self.training
            else getattr(self.args, "batch_size_valid", 0)
        )
        if expected_bsz is None:
            expected_bsz = 0

        if (
            not self.in_generation
            and expected_bsz != 0
            and input_shape[0] != expected_bsz
        ):
            print(
                f"Warning: padding batch with unexpected size {input_shape[0]} (expected: {expected_bsz})"
            )
            assert input_shape[0] < expected_bsz

            padded_input = torch.zeros(
                (expected_bsz, input_shape[1], input_shape[2]),
                dtype=input_tensor.dtype,
                layout=input_tensor.layout,
                device=input_tensor.device,
            )
            padded_input[: input_shape[0], :, :] = input_tensor
            input_tensor = padded_input

            padded_input_padding_mask = torch.ones(
                (expected_bsz, input_shape[1]),
                dtype=torch.bool,
                device=input_tensor.device,
            )
            if input_padding_mask is not None:
                padded_input_padding_mask[: input_shape[0], :] = (
                    input_padding_mask
                )
            else:
                padded_input_padding_mask[: input_shape[0], :] = False
            input_padding_mask = padded_input_padding_mask

        # Reshape into S tokens by dropping sequence dimension
        reshaped_input = input_tensor.reshape(-1, dim)
        reshaped_input_shape = reshaped_input.shape
        reshaped_input_padding_mask = (
            input_padding_mask.reshape(-1)
            if input_padding_mask is not None
            else None
        )

        # Handle max-tokens padding
        if expected_bsz == 0:
            expected_dim = reshaped_input_shape[0] * torch.ones(
                (1,), dtype=torch.long, device=input_tensor.device
            )
            if dist.is_initialized():
                dist.all_reduce(
                    expected_dim, group=dist.group.WORLD, op=dist.ReduceOp.MAX
                )
            expected_dim = int(expected_dim.item())

            padded_input = torch.zeros(
                (expected_dim, reshaped_input_shape[1]),
                dtype=input_tensor.dtype,
                layout=input_tensor.layout,
                device=input_tensor.device,
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

        # Apply gating
        l_aux, combine_weights, dispatch_mask, self.metadata = self.gate(
            reshaped_input, reshaped_input_padding_mask
        )

        dispatch_mask = dispatch_mask.to(input_tensor.dtype).permute(
            1, 2, 0
        )  # S,E,C -> E,C,S
        E, C, S = dispatch_mask.size()
        M = reshaped_input.size(1)
        assert reshaped_input.size() == (S, M)

        # Dispatch tokens to experts
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
            expert_outputs.append(expert(chunk))
        expert_output = torch.cat(expert_outputs, dim=1)

        if self.all2all_size > 1:
            expert_output = self.all_to_all_wrapper(expert_output)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(
            self.all2all_size * self.num_local_experts, -1, dim
        )

        # Combine expert outputs
        combined_output = combine_weights.view(S, E * C).mm(
            expert_output.view(E * C, M)
        )

        # Remove padding
        combined_output = combined_output[: reshaped_input_shape[0], :]
        combined_output = combined_output.reshape(input_tensor.shape)
        combined_output = combined_output[: input_shape[0], :, :]

        self.record_all_to_all_stats()

        return combined_output, l_aux

    def prepare_for_inference_(self) -> None:
        """Prepare the MoE layer for inference mode."""
        self.in_generation = True

    def all_to_all_wrapper(self, input_tensor: Tensor) -> Tensor:
        """Wrapper function for all-to-all communication."""
        dummy_a2a = getattr(self.args, "dummy_a2a", False)
        if dummy_a2a:
            input_tensor = input_tensor.contiguous()
            return input_tensor.detach().clone()

        # Record timing
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cpu_start = time.time() * 1000
        cuda_start.record()

        output = _AllToAll.apply(self.all2all_group, input_tensor)

        cuda_end.record()
        cpu_end = time.time() * 1000
        self.a2a_cpu_time_ms += cpu_end - cpu_start
        self.a2a_cuda_event_intervals.append((cuda_start, cuda_end))
        return output

    def record_all_to_all_stats(self) -> None:
        """Record statistics for all-to-all communication performance."""
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
        # Reset stats
        self.a2a_cpu_time_ms = 0.0
        self.a2a_cuda_event_intervals = []
