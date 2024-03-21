from torch import nn


def text_to_tensor(
    text: str,
    tokenizer: callable,
    process_func: callable,
    dim: int,
    num_tokens: int,
):
    """
    Converts a given text into a tensor representation.

    Args:
        text (str): The input text to be converted.
        tokenizer (callable): A callable object that tokenizes the text.
        process_func (callable): A callable object that processes the tokens.
        dim (int): The dimension of the embedding.
        num_tokens (int): The number of tokens in the vocabulary.

    Returns:
        out: The tensor representation of the input text.
    """
    tokens = tokenizer(text)

    # Truncate or pad the tokens to the specified length
    tokens = process_func(tokens)

    # Convert the tokens to a tensor
    out = nn.Embedding(num_tokens, dim)(tokens)
    return out
