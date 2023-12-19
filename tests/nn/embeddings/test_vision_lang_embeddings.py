import pytest
import torch
from torch import nn
from zeta.nn.embeddings.vis_lang_emb import VisionLanguageEmbedding


# Test case for default initialization
def test_default_init():
    text_embed = nn.Embedding(10, 10)
    vision_embed = nn.Embedding(10, 10)
    module = VisionLanguageEmbedding(text_embed, vision_embed)
    assert isinstance(module.text_embed, nn.Module)
    assert isinstance(module.vision_embed, nn.Module)


# Test case for forward pass with text input only
def test_forward_text_input():
    text_embed = nn.Embedding(10, 10)
    vision_embed = nn.Embedding(10, 10)
    module = VisionLanguageEmbedding(text_embed, vision_embed)
    textual_tokens = torch.randint(0, 10, (10,))
    y = module(textual_tokens, None)
    assert y.shape == (10, 10)


# Test case for forward pass with vision input only
def test_forward_vision_input():
    text_embed = nn.Embedding(10, 10)
    vision_embed = nn.Embedding(10, 10)
    module = VisionLanguageEmbedding(text_embed, vision_embed)
    visual_tokens = torch.randint(0, 10, (10,))
    y = module(None, visual_tokens)
    assert y.shape == (10, 10)


# Test case for forward pass with both text and vision inputs
def test_forward_both_inputs():
    text_embed = nn.Embedding(10, 10)
    vision_embed = nn.Embedding(10, 10)
    module = VisionLanguageEmbedding(text_embed, vision_embed)
    textual_tokens = torch.randint(0, 10, (10,))
    visual_tokens = torch.randint(0, 10, (10,))
    y = module(textual_tokens, visual_tokens)
    assert y.shape == (10, 20)


# Test case for initializing with incorrect text embedding
def test_incorrect_text_embedding_init():
    text_embed = nn.Linear(10, 10)
    vision_embed = nn.Embedding(10, 10)
    with pytest.raises(AssertionError):
        VisionLanguageEmbedding(text_embed, vision_embed)


# Test case for initializing with incorrect vision embedding
def test_incorrect_vision_embedding_init():
    text_embed = nn.Embedding(10, 10)
    vision_embed = nn.Linear(10, 10)
    with pytest.raises(AssertionError):
        VisionLanguageEmbedding(text_embed, vision_embed)


# Test case for forward pass with text input being None
def test_forward_text_input_none():
    text_embed = nn.Embedding(10, 10)
    vision_embed = nn.Embedding(10, 10)
    module = VisionLanguageEmbedding(text_embed, vision_embed)
    visual_tokens = torch.randint(0, 10, (10,))
    y = module(None, visual_tokens)
    assert y.shape == (10, 10)


# Test case for forward pass with vision input being None
def test_forward_vision_input_none():
    text_embed = nn.Embedding(10, 10)
    vision_embed = nn.Embedding(10, 10)
    module = VisionLanguageEmbedding(text_embed, vision_embed)
    textual_tokens = torch.randint(0, 10, (10,))
    y = module(textual_tokens, None)
    assert y.shape == (10, 10)
