import os
from zeta.tokenizers.sentence_piece import SentencePieceTokenizer


def test_sentence_piece_tokenizer_initialization():
    model_path = "/path/to/your/model"  # replace with your actual model path
    assert os.path.isfile(model_path), "Model file does not exist"

    tokenizer = SentencePieceTokenizer(model_path)

    assert isinstance(tokenizer, SentencePieceTokenizer)
    assert tokenizer.n_words == tokenizer.sp_model.vocab_size()
    assert tokenizer.bos_id == tokenizer.sp_model.bos_id()
    assert tokenizer.eos_id == tokenizer.sp_model.eos_id()
    assert tokenizer.pad_id == tokenizer.sp_model.pad_id()


def test_sentence_piece_tokenizer_encode():
    model_path = "/path/to/your/model"  # replace with your actual model path
    tokenizer = SentencePieceTokenizer(model_path)

    text = "Hello, world!"
    encoded_text = tokenizer.encode(text, bos=True, eos=True)

    assert isinstance(encoded_text, list)
    assert encoded_text[0] == tokenizer.bos_id
    assert encoded_text[-1] == tokenizer.eos_id


def test_sentence_piece_tokenizer_decode():
    model_path = "/path/to/your/model"  # replace with your actual model path
    tokenizer = SentencePieceTokenizer(model_path)

    text = "Hello, world!"
    encoded_text = tokenizer.encode(text, bos=True, eos=True)
    decoded_text = tokenizer.decode(encoded_text)

    assert isinstance(decoded_text, str)
    assert decoded_text == text


def test_sentence_piece_tokenizer_encode_infilling():
    model_path = "/path/to/your/model"  # replace with your actual model path
    tokenizer = SentencePieceTokenizer(model_path)

    text = "Hello, world!"
    encoded_text = tokenizer.encode_infilling(text)

    assert isinstance(encoded_text, list)


def test_sentence_piece_tokenizer_decode_infilling():
    model_path = "/path/to/your/model"  # replace with your actual model path
    tokenizer = SentencePieceTokenizer(model_path)

    text = "Hello, world!"
    encoded_text = tokenizer.encode_infilling(text)
    decoded_text = tokenizer.decode_infilling(encoded_text)

    assert isinstance(decoded_text, str)
    assert (decoded_text == text[1:]
           )  # the first character is removed in decode_infilling
