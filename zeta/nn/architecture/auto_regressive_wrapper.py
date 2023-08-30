import torch
import torch.nn.functional as F
from einops import pack, rearrange, unpack
from torch import nn
from zeta.utils.helpers import (  # noqa: E402
    eval_decorator,
    exists,
    once,  # noqa: F401
    
)
from zeta.utils.inference_helpers import top_a, top_k, top_p

class AutoregressiveWrapper(nn.Module):
    def __init__(
        self,
        net,
        ignore_index = -100,
        pad_value = 0,
        mask_prob = 0.
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

        # paper shows masking (MLM) in conjunction with autoregressive decoder-only training leads to big improvements https://arxiv.org/abs/2210.13432
        assert mask_prob < 1.
        self.mask_prob = mask_prob

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        start_tokens,
        seq_len,
        eos_token = None,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_thres = 0.9,
        min_p_pow = 2.0,
        min_p_ratio = 0.02,
        **kwargs
    ):

        start_tokens, ps = pack([start_tokens], '* n')

        b, t = start_tokens.shape

        out = start_tokens

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]

            logits = self.net(x, **kwargs)[:, -1]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres = filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            elif filter_logits_fn is top_a:
                filtered_logits = filter_logits_fn(logits, min_p_pow = min_p_pow, min_p_ratio= min_p_ratio)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if exists(eos_token):
                is_eos_tokens = (out == eos_token)

                if is_eos_tokens.any(dim = -1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
                    out = out.masked_fill(mask, self.pad_value)
                    break

        out = out[:, t:]

        out, = unpack(out, ps, '* n')

        return out

    def forward(self, x, return_loss=True, **kwargs):
        seq, ignore_index = x.shape[1], self.ignore_index

        inp, target = x[:, :-1], x[:, 1:]

        if self.mask_prob > 0.:
            rand = torch.randn(inp.shape, device = x.device)
            rand[:, 0] = -torch.finfo(rand.dtype).max # first token should not be masked out
            num_mask = min(int(seq * self.mask_prob), seq - 1)
            indices = rand.topk(num_mask, dim = -1).indices
            mask = ~torch.zeros_like(inp).scatter(1, indices, 1.).bool()
            kwargs.update(self_attn_context_mask = mask)

        logits = self.net(inp, **kwargs)

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = ignore_index
        )

        if return_loss:
            return logits, loss

        return logits
