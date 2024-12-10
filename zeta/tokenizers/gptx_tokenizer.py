from transformers import AutoTokenizer


class LanguageTokenizerGPTX:
    """
    LanguageTokenizerGPTX is a class that provides tokenization and decoding functionality using the GPT-Neox-20B model.
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b",
            eos_token="<eos>",
            pad_token="<pad>",
            extra_ids=0,
            model_max_length=8192,
        )

    def tokenize_texts(self, texts):
        """
        Tokenizes a list of texts using the GPT-Neox-20B tokenizer.

        Args:
            texts (List[str]): A list of texts to be tokenized.

        Returns:
            torch.Tensor: The tokenized input IDs as a PyTorch tensor.
        """
        return self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).input_ids

    def decode(self, texts):
        """
        Decodes a list of tokenized input IDs into text.

        Args:
            texts (torch.Tensor): The tokenized input IDs as a PyTorch tensor.

        Returns:
            str: The decoded text.
        """
        return self.tokenizer.decode(texts)

    def __len__(self):
        """
        Returns the number of tokens in the tokenizer's vocabulary.

        Returns:
            int: The number of tokens in the vocabulary.
        """
        num_tokens = len(self.tokenizer)
        return num_tokens
