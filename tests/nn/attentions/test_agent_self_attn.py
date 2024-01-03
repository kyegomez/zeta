import torch
from torch import nn
from zeta.nn.attention.agent_attn import AgentSelfAttention


def test_agent_self_attention_init():
    agent_self_attn = AgentSelfAttention(dim=64, num_agent_tokens=16)
    assert isinstance(agent_self_attn, AgentSelfAttention)
    assert agent_self_attn.scale == 64**-0.5
    assert isinstance(agent_self_attn.to_qkv, nn.Sequential)
    assert isinstance(agent_self_attn.to_gates, nn.Sequential)
    assert isinstance(agent_self_attn.agent_tokens, nn.Parameter)
    assert isinstance(agent_self_attn.qa_talking_heads, nn.Conv2d)
    assert isinstance(agent_self_attn.ak_talking_heads, nn.Conv2d)
    assert isinstance(agent_self_attn.qa_dropout, nn.Dropout)
    assert isinstance(agent_self_attn.ak_dropout, nn.Dropout)
    assert isinstance(agent_self_attn.to_out, nn.Sequential)


def test_agent_self_attention_forward():
    agent_self_attn = AgentSelfAttention(dim=64, num_agent_tokens=16)
    x = torch.randn(2, 64)
    output = agent_self_attn(x)
    assert output.shape == x.shape


def test_agent_self_attention_forward_with_mask():
    agent_self_attn = AgentSelfAttention(dim=64, num_agent_tokens=16)
    x = torch.randn(2, 64)
    mask = torch.ones(2, 64).bool()
    output = agent_self_attn(x, mask=mask)
    assert output.shape == x.shape


def test_agent_self_attention_forward_with_agent_tokens():
    agent_self_attn = AgentSelfAttention(dim=64, num_agent_tokens=16)
    x = torch.randn(2, 64)
    agent_tokens = torch.randn(2, 8, 16, 64)
    output, agent_gathered_tokens = agent_self_attn(
        x, agent_tokens=agent_tokens, return_agent_tokens=True
    )
    assert output.shape == x.shape
    assert agent_gathered_tokens.shape == agent_tokens.shape
