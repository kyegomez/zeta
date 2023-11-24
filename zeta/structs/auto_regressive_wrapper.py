import torch
import torch.nn.functional as F
from einops import pack, rearrange, unpack
from torch import nn

from zeta.utils.main import (  # noqa: E402
    eval_decorator,
    exists,
    once,  # noqa: F401
    top_a,
    top_k,
    top_p,
)


# Utils
def temperature_sampling(self, logits, temperature):
    return torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)


def top_p_sampling(self, logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
        ..., :-1
    ].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float("-inf")
    return torch.multinomial(F.softmax(logits, dim=-1), 1)


def classifier_free_guidance(self, logits_cond, logits_uncond, alpha):
    return logits_uncond + alpha * (logits_cond - logits_uncond)


def contrastive_guidance(self, logits, k):
    top_k_logits, _ = torch.topk(logits, k)
    return torch.multinomial(F.softmax(top_k_logits, dim=-1), 1)


class AutoregressiveWrapper(nn.Module):
    def __init__(
        self,
        net,
        ignore_index=-100,
        pad_value=0,
        mask_prob=0.0,
        speculative=False,
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

        # paper shows masking (MLM) in conjunction with autoregressive
        # decoder-only training leads to big improvements
        # https://arxiv.org/abs/2210.13432
        assert mask_prob < 1.0
        self.mask_prob = mask_prob

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        start_tokens,
        seq_len,
        eos_token=None,
        strategy="temperature",
        temperature=1.0,
        filter_logits_fn=top_k,
        filter_thres=0.9,
        min_p_pow=2.0,
        min_p_ratio=0.02,
        gamma=5,  # number of guesses for speculative decoding
        **kwargs,
    ):
        start_tokens, ps = pack([start_tokens], "* n")

        b, t = start_tokens.shape

        out = start_tokens

        if self.speculative:
            for _ in range(seq_len):
                x = out[:, -self.max_seq_len]
                logits = self.net(x, **kwargs)[:, -1]

                if filter_logits_fn in {top_k, top_p}:
                    filtered_logits = filter_logits_fn(
                        logits, thres=filter_thres
                    )
                    probs = F.softmax(filtered_logits / temperature, dim=-1)
                elif filter_logits_fn is top_a:
                    filtered_logits = filter_logits_fn(
                        logits, min_p_pow=min_p_pow, min_p_ratio=min_p_ratio
                    )
                    probs = F.softmax(filtered_logits / temperature, dim=-1)

                # speculative decoding
                guesses = torch.multinomial(probs, gamma, replacement=True)

                p_values = []
                for guess in guesses:
                    x_prime = torch.cat((x, guess.unsqueeze(0)), dim=1)
                    logits_prime = self.net(x_prime, **kwargs)[:, -1]
                    p_values.append(
                        F.softmax(logits_prime / temperature, dim=-1)
                    )

                n = gamma
                for i in range(gamma):
                    ri = torch.rand(1).item()
                    if (
                        ri
                        > p_values[i][guesses[i].item()]
                        / probs[guesses[i].item()]
                    ):
                        n = i - 1
                        break

                p_0 = p_values[n]
                if n < gamma:
                    q_n = probs[guesses[n].item()]
                    p_0 = F.normalize(torch.clamp(p_0 - q_n, min=0), p=1, dim=0)

                sample = torch.multinomial(p_0, 1)

                out = torch.cat((out, sample), dim=-1)

                if exists(eos_token):
                    is_eos_tokens = out == eos_token

                    if is_eos_tokens.any(dim=-1).all():
                        # mask out everything after the eos tokens
                        shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                        mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                        out = out.masked_fill(mask, self.pad_value)
                        break

                out = out[:, t:]
                (out,) = unpack(out, ps, "* n")
                return out
        else:
            for _ in range(seq_len):
                x = out[:, -self.max_seq_len :]

                logits = self.net(x, **kwargs)[:, -1]

                if filter_logits_fn in {top_k, top_p}:
                    filtered_logits = filter_logits_fn(
                        logits, thres=filter_thres
                    )
                    probs = F.softmax(filtered_logits / temperature, dim=-1)

                elif filter_logits_fn is top_a:
                    filtered_logits = filter_logits_fn(
                        logits, min_p_pow=min_p_pow, min_p_ratio=min_p_ratio
                    )
                    probs = F.softmax(filtered_logits / temperature, dim=-1)

                sample = torch.multinomial(probs, 1)

                out = torch.cat((out, sample), dim=-1)

                if exists(eos_token):
                    is_eos_tokens = out == eos_token

                    if is_eos_tokens.any(dim=-1).all():
                        # mask out everything after the eos tokens
                        shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                        mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                        out = out.masked_fill(mask, self.pad_value)
                        break

            out = out[:, t:]

            (out,) = unpack(out, ps, "* n")

            return out

    def forward(self, x, return_loss=True, **kwargs):
        seq, ignore_index = x.shape[1], self.ignore_index

        inp, target = x[:, :-1], x[:, 1:]

        if self.mask_prob > 0.0:
            rand = torch.randn(inp.shape, device=x.device)
            # first token should not be masked out
            rand[:, 0] = -torch.finfo(rand.dtype).max
            num_mask = min(int(seq * self.mask_prob), seq - 1)
            indices = rand.topk(num_mask, dim=-1).indices
            mask = ~torch.zeros_like(inp).scatter(1, indices, 1.0).bool()
            kwargs.update(self_attn_context_mask=mask)

        logits = self.net(inp, **kwargs)

        loss = F.cross_entropy(
            rearrange(logits, "b n c -> b c n"),
            target,
            ignore_index=ignore_index,
        )

        if return_loss:
            return logits, loss

        return logits
