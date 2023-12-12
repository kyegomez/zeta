import pytest
import os
from zeta.tokenizers.llama_sentencepiece import LLamaTokenizer


def test_llama_tokenizer_init_model_path():
    model_path = "/path/to/model"
    tokenizer = LLamaTokenizer(model_path=model_path)
    assert tokenizer.sp_model is not None


def test_llama_tokenizer_init_tokenizer_name():
    tokenizer_name = "hf-internal-testing/llama-tokenizer"
    tokenizer = LLamaTokenizer(tokenizer_name=tokenizer_name)
    assert tokenizer.sp_model is not None


def test_llama_tokenizer_init_no_args():
    with pytest.raises(ValueError):
        LLamaTokenizer()


def test_llama_tokenizer_encode():
    model_path = "/path/to/model"
    tokenizer = LLamaTokenizer(model_path=model_path)
    encoded_text = tokenizer.encode("This is a sample text")
    assert isinstance(encoded_text, list)
    assert all(isinstance(i, int) for i in encoded_text)


def test_llama_tokenizer_decode():
    model_path = "/path/to/model"
    tokenizer = LLamaTokenizer(model_path=model_path)
    decoded_text = tokenizer.decode([1, 2, 3])
    assert isinstance(decoded_text, str)


@pytest.mark.parametrize("text", ["", " ", "  ", "\t", "\n"])
def test_llama_tokenizer_encode_empty(text):
    model_path = "/path/to/model"
    tokenizer = LLamaTokenizer(model_path=model_path)
    encoded_text = tokenizer.encode(text)
    assert encoded_text == []


@pytest.mark.parametrize("ids", [[], [0], [0, 1], [0, 1, 2]])
def test_llama_tokenizer_decode_empty(ids):
    model_path = "/path/to/model"
    tokenizer = LLamaTokenizer(model_path=model_path)
    decoded_text = tokenizer.decode(ids)
    assert isinstance(decoded_text, str)


@pytest.mark.parametrize(
    "text",
    ["This is a sample text", "Another sample text", "Yet another sample text"],
)
def test_llama_tokenizer_encode_decode(text):
    model_path = "/path/to/model"
    tokenizer = LLamaTokenizer(model_path=model_path)
    encoded_text = tokenizer.encode(text)
    decoded_text = tokenizer.decode(encoded_text)
    assert text == decoded_text


@pytest.mark.parametrize(
    "tokenizer_name",
    [
        "hf-internal-testing/llama-tokenizer",
        "another-tokenizer",
        "yet-another-tokenizer",
    ],
)
def test_llama_tokenizer_download_tokenizer(tokenizer_name):
    tokenizer = LLamaTokenizer(tokenizer_name=tokenizer_name)
    assert os.path.isfile("data/tokenizer.model")
