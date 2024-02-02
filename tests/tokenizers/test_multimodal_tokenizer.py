from PIL import Image
import torch
from zeta.tokenizers.multi_modal_tokenizer import MultiModalTokenizer


def test_multi_modal_tokenizer_initialization():
    tokenizer = MultiModalTokenizer()

    assert isinstance(tokenizer, MultiModalTokenizer)
    assert tokenizer.max_length == 8192
    assert tokenizer.tokenizer.eos_token == "<eos>"
    assert tokenizer.tokenizer.pad_token == "<pad>"
    assert tokenizer.tokenizer.model_max_length == tokenizer.max_length
    assert tokenizer.im_idx == tokenizer.tokenizer.convert_tokens_to_ids(
        "<image>")
    assert tokenizer.im_end_idx == tokenizer.tokenizer.convert_tokens_to_ids(
        "</image>")


def test_multi_modal_tokenizer_tokenize_texts():
    tokenizer = MultiModalTokenizer()

    texts = ["Hello, world!", "Goodbye, world!"]
    tokenized_texts, only_text_tokens = tokenizer.tokenize_texts(texts)

    assert isinstance(tokenized_texts, torch.Tensor)
    assert tokenized_texts.shape[0] == len(texts)
    assert isinstance(only_text_tokens, torch.Tensor)
    assert only_text_tokens.shape[0] == len(texts)


def test_multi_modal_tokenizer_tokenize_images():
    tokenizer = MultiModalTokenizer()

    # Assuming images is a list of PIL Image objects
    images = [Image.new("RGB", (60, 30), color="red") for _ in range(2)]
    tokenized_images = tokenizer.tokenize_images(images)

    assert isinstance(tokenized_images, torch.Tensor)
    assert tokenized_images.shape[0] == len(images)


def test_multi_modal_tokenizer_tokenize():
    tokenizer = MultiModalTokenizer()

    sample = {
        "target_text": ["Hello, world!", "Goodbye, world!"],
        "image": [Image.new("RGB", (60, 30), color="red") for _ in range(2)],
    }
    tokenized_sample = tokenizer.tokenize(sample)

    assert isinstance(tokenized_sample, dict)
    assert "text_tokens" in tokenized_sample
    assert "images" in tokenized_sample
    assert "labels" in tokenized_sample
    assert "attention_mask" in tokenized_sample
