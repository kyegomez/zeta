import torch
from torch import nn, Tensor
from copy import deepcopy
import torch.nn.functional as F
from einops import rearrange


def freeze_all_layers(module):
    for param in module.parameters():
        param.reqires_grad = False


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def log_prob(prob, indices, eps=1e-20):
    indices = rearrange(indices, "... -> ... 1")
    log_probs = log(prob.gather(-1, indices), eps=eps)
    return rearrange(log_probs, "... 1 -> ...")


def log_prob_from_model_and_seq(model, seq):
    logits = model(seq)
    prob = logits.softmax(dim=-1)
    return log_prob(prob, seq)


class DPO(nn.Module):
    """
    Deep Policy Optimization (DPO) module.

    Args:
        model (nn.Module): The policy model.
        beta (float, optional): The beta parameter. Defaults to 0.1.
    """

    def __init__(self, model: nn.Module, *, beta: float = 0.1):
        super().__init__()
        self.policy_model = model

        self.ref_model = deepcopy(model)
        freeze_all_layers(self.ref_model)

        self.beta = beta

    def parameters(self):
        return self.policy_model.parameters()

    def forward(self, preferred_seq: Tensor, unpreferred_seq: Tensor):
        """
        Forward pass of the DPO module.

        Args:
            preferred_seq (torch.Tensor): The preferred sequence.
            unpreferred_seq (torch.Tensor): The unpreferred sequence.

        Returns:
            torch.Tensor: The loss value.
        """
        assert preferred_seq.ndim == 2
        assert preferred_seq.shape == unpreferred_seq.shape

        """
        Following Appendix B in https://arxiv.org/abs/2305.18290
        """

        with torch.no_grad():
            self.ref_model.eval()
            ref_preferred_logprob = log_prob_from_model_and_seq(
                self.ref_model, preferred_seq
            )
            ref_unpreferred_logprob = log_prob_from_model_and_seq(
                self.ref_model, unpreferred_seq
            )

        policy_preferred_logprob = log_prob_from_model_and_seq(
            self.policy_model, preferred_seq
        )
        policy_unpreferred_logprob = log_prob_from_model_and_seq(
            self.policy_model, unpreferred_seq
        )

        policy_logratios = policy_preferred_logprob - policy_unpreferred_logprob
        ref_logratios = ref_preferred_logprob - ref_unpreferred_logprob

        losses = -F.logsigmoid(self.beta * (policy_logratios - ref_logratios))

        return losses.mean()
