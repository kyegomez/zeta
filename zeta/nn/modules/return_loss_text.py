import torch
from einops import rearrange
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from zeta.structs.auto_regressive_wrapper import AutoRegressiveWrapper
from typing import List
from einops import reduce


def exists(val):
    return val is not None


def return_loss_text(
    x: Tensor, logits: Tensor, labels: Tensor, ignore_index, mask: Tensor
):
    """
    Computes the cross-entropy loss between the predicted logits and the target labels.

    Args:
        logits (Tensor): The predicted logits of shape (batch_size, num_classes, sequence_length).
        labels (Tensor): The target labels of shape (batch_size, sequence_length).
        ignore_index (int): The index to ignore when computing the loss.

    Returns:
        Tensor: The computed cross-entropy loss.
    """
    seq, labels = x[:, :-1], x[:, 1:]

    labels = labels.masked_fill(~mask[:, 1:], ignore_index)

    loss = F.cross_entropy(
        rearrange(logits, "b n c -> b c n"), labels, ignore_index=ignore_index
    )

    return loss


def add_masking_llm(x: Tensor, mask: Tensor, ignore_index: int):
    """
    Adds masking to the input tensor.

    Args:
        x (Tensor): The input tensor.
        ignore_index (int): The index to ignore.

    Returns:
        Tensor: The masked input tensor.
    """
    ...


def calc_z_loss(
    pre_softmax_attns: List[Tensor], mask: Tensor = None, weight: float = 1.0
):
    lse = 0.0

    for attn in pre_softmax_attns:
        lse = lse + attn.logsumexp(dim=-1)

    loss = torch.square(lse)
    loss = reduce(loss, "b h n -> b n", "sum")

    if not exists(mask):
        return loss.mean() * weight

    loss = loss[mask].sum() / mask.sum().clamp(min=1e-5)
    return loss * weight


def max_neg_value(tensor: Tensor):
    return -torch.finfo(tensor.dtype).max


def l2norm(x: Tensor, groups: int = 1):
    """
    Applies L2 normalization to the input tensor.

    Args:
        x (Tensor): The input tensor to be normalized.
        groups (int, optional): The number of groups to divide the input tensor into. Defaults to 1.

    Returns:
        Tensor: The normalized tensor.

    """
    x = rearrange(x, "... (g d) -> ... g d", g=groups)
    x = F.normalize(x, p=2, dim=-1)
    return rearrange(x, "... g d -> ... (g d)")


class TextTokenEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        num_tokens: int,
        l2norm_embed: bool = True,
    ):
        """
        Initializes a TextTokenEmbedding module.

        Args:
            dim (int): The dimension of the embedding.
            num_tokens (int): The number of tokens in the vocabulary.
            l2norm_embed (bool, optional): Whether to apply L2 normalization to the embeddings. Defaults to True.
        """
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.l2norm_embed = l2norm_embed
        self.embed = nn.Embedding(num_tokens, dim)

    def forward(self, x: Tensor):
        """
        Forward pass of the TextTokenEmbedding module.

        Args:
            x (Tensor): The input tensor of shape (batch_size, sequence_length).

        Returns:
            Tensor: The embedded tensor of shape (batch_size, sequence_length, dim).
        """
        token_embed = self.embed(x.long())
        return l2norm(token_embed) if self.l2norm_embed else token_embed


def dropout_seq(seq: Tensor, mask: Tensor, dropout: float = 0.0):
    """
    Applies dropout to a sequence of tensors.

    Args:
        seq (Tensor): The input sequence tensor of shape (batch_size, sequence_length, ...).
        mask (Tensor): The mask tensor of shape (batch_size, sequence_length) indicating which elements to keep.
        dropout (float, optional): The dropout probability. Defaults to 0.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the modified sequence tensor and the modified mask tensor.

    """
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device=device)

    if exists(mask):
        mask_value = max_neg_value(logits)
        logits = logits.masked_fill(~mask, mask_value)

    keep_prob = 1.0 - dropout
    num_keep = max(1, int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim=1).indices

    batch_indices = torch.arange(b, device=device)
    batch_indices = rearrange(batch_indices, "b -> b 1")

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim=-1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device=device) < rearrange(
            seq_keep_counts, "b -> b 1"
        )

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask


@torch.no_grad()
def transformer_generate(
    model: nn.Module,
    prompt: Tensor,
    temperature: float = 0.5,
    filter_threshold: float = 0.9,
    *args,
    **kwargs,
):
    """
    Generates text given a prompt.

    Args:
        model (nn.Module): The model to generate text.
        prompt (Tensor): The prompt tensor.

    Returns:
        Tensor: The generated text.
    """
    model = AutoRegressiveWrapper(net=model)

    return model.generate(
        prompt,
        filter_thres=filter_threshold,
        temperature=temperature,
        *args,
        **kwargs,
    )
