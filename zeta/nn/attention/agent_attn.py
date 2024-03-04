import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
from torch.nn import Module

# functions


def exists(v):
    return v is not None


# main class


class AgentSelfAttention(Module):
    """
    Self-attention module for agent tokens in a neural network.

    Args:
        dim (int): The input dimension.
        num_agent_tokens (int): The number of agent tokens.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        heads (int, optional): The number of attention heads. Defaults to 8.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        talking_heads (bool, optional): Whether to use talking heads mechanism. Defaults to True.
        gate (bool, optional): Whether to apply gating mechanism. Defaults to True.
        combine_agent_tokens (bool, optional): Whether to combine agent tokens. Defaults to False.

    Examples::
        >>> import torch
        >>> from zeta.nn.attention import AgentSelfAttention
        >>> agent_self_attn = AgentSelfAttention(dim=64, num_agent_tokens=16)
        >>> x = torch.randn(2, 64)
        >>> output = agent_self_attn(x)
        >>> output.shape
        torch.Size([2, 64])
    """

    def __init__(
        self,
        dim,
        *,
        num_agent_tokens,
        dim_head=64,
        heads=8,
        dropout=0.1,
        talking_heads=True,
        gate=True,
        combine_agent_tokens=False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        dim_inner = dim_head * heads

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h n d", h=heads, qkv=3),
        )

        self.to_gates = (
            nn.Sequential(
                nn.Linear(dim, heads),
                Rearrange("b n h -> b h n 1"),
                nn.Sigmoid(),
            )
            if gate
            else None
        )

        self.agent_tokens = nn.Parameter(
            torch.zeros(heads, num_agent_tokens, dim_head)
        )
        nn.init.normal_(self.agent_tokens, std=0.02)

        self.qa_talking_heads = (
            nn.Conv2d(heads, heads, 1, bias=False)
            if talking_heads
            else nn.Identity()
        )
        self.ak_talking_heads = (
            nn.Conv2d(heads, heads, 1, bias=False)
            if talking_heads
            else nn.Identity()
        )

        self.qa_dropout = nn.Dropout(dropout)
        self.ak_dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            Rearrange("b h n d -> b n (h d)"),
            nn.Linear(dim_inner, dim, bias=False),
        )

    def forward(
        self, x, mask=None, agent_tokens=None, return_agent_tokens=False
    ):
        batch = x.shape[0]

        q, k, v = self.to_qkv(x)

        if exists(agent_tokens):
            a = agent_tokens
        else:
            a = repeat(self.agent_tokens, "h m d -> b h m d", b=batch)

        a = a * self.scale

        qa_sim = einsum("b h i d, b h j d -> b h i j", q, a)
        ak_sim = einsum("b h i d, b h j d -> b h i j", a, k)

        if exists(mask):
            max_neg_value = -torch.finfo(qa_sim.dtype).max
            ak_sim = ak_sim.masked_fill(
                ~rearrange(mask, "b j -> b 1 1 j"), max_neg_value
            )

        qa_attn = qa_sim.softmax(dim=-1)
        ak_attn = ak_sim.softmax(dim=-1)

        qa_attn = self.qa_dropout(qa_attn)
        ak_attn = self.ak_dropout(ak_attn)

        qa_attn = self.qa_talking_heads(qa_attn)
        ak_attn = self.ak_talking_heads(ak_attn)

        agent_gathered_tokens = einsum(
            "b h i j, b h j d -> b h i d", ak_attn, v
        )

        out = einsum(
            "b h i j, b h j d -> b h i d", qa_attn, agent_gathered_tokens
        )

        if exists(mask):
            out = out.masked_fill(~rearrange(mask, "b n -> b 1 n 1"), 0.0)

        if exists(self.to_gates):
            out = out * self.to_gates(x)

        out = self.to_out(out)

        if not return_agent_tokens:
            return out

        return out, agent_gathered_tokens
