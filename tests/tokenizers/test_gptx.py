import torch
import pytest
from zeta.tokenizers.gptx_tokenizer import LanguageTokenizerGPTX


def test_language_tokenizer_gptx_initialization():
    tokenizer = LanguageTokenizerGPTX()

    assert isinstance(tokenizer, LanguageTokenizerGPTX)
    assert tokenizer.tokenizer.eos_token == "<eos>"
    assert tokenizer.tokenizer.pad_token == "<pad>"
    assert tokenizer.tokenizer.model_max_length == 8192


def test_language_tokenizer_gptx_tokenize_texts():
    tokenizer = LanguageTokenizerGPTX()

    texts = ["Hello, world!", "Goodbye, world!"]
    tokenized_texts = tokenizer.tokenize_texts(texts)

    assert isinstance(tokenized_texts, torch.Tensor)
    assert tokenized_texts.shape[0] == len(texts)


def test_language_tokenizer_gptx_decode():
    tokenizer = LanguageTokenizerGPTX()

    texts = ["Hello, world!", "Goodbye, world!"]
    tokenized_texts = tokenizer.tokenize_texts(texts)
    decoded_texts = tokenizer.decode(tokenized_texts[0])

    assert isinstance(decoded_texts, str)


def test_language_tokenizer_gptx_len():
    tokenizer = LanguageTokenizerGPTX()

    num_tokens = len(tokenizer)

    assert isinstance(num_tokens, int)
    assert num_tokens > 0
