# Using LLAMA tokenizer
import os
from logging import getLogger

import requests
from sentencepiece import SentencePieceProcessor

logger = getLogger()

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model",
    },
    "tokenizer_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer_config.json",
    },
}


class LLamaTokenizer:
    """
    A tokenizer that uses a pretrained SentencePiece model for text tokenization.

    Args:
        model_path: Path to a pretrained SentencePiece model file.
        tokenizer_name: Name of a pretrained SentencePiece model hosted on HuggingFace Hub.

    Examples:
        >>> tokenizer_name = "hf-internal-testing/llama-tokenizer"
        >>> tokenizer = Tokenizer(tokenizer_name=tokenizer_name)
        >>> encoded_text = tokenizer.encode("This is a sample text")
        >>> decoded_text = tokenizer.decode(encoded_text)
        >>> print("Encoded text:", encoded_text)
        >>> print("Decoded text:", decoded_text)
    """

    def __init__(self, model_path: str = None, tokenizer_name: str = None):
        if model_path:
            assert os.path.isfile(model_path), model_path
        elif tokenizer_name:
            model_path = self.download_tokenizer(tokenizer_name)
        else:
            raise ValueError(
                "Either model_path or tokenizer_name must be provided."
            )

        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

    @staticmethod
    def download_tokenizer(tokenizer_name: str) -> str:
        if tokenizer_name not in PRETRAINED_VOCAB_FILES_MAP["vocab_file"]:
            raise ValueError(f"Tokenizer {tokenizer_name} is not available.")

        model_url = PRETRAINED_VOCAB_FILES_MAP["vocab_file"][tokenizer_name]
        model_path = os.path.join("data", "tokenizer.model")

        if not os.path.exists("data"):
            os.makedirs("data")

        # Downloading the tokenizer model file
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, "wb") as file:
                file.write(response.content)
            logger.info(f"Downloaded SentencePiece model to {model_path}")
        else:
            raise Exception(f"Failed to download model from {model_url}")

        return model_path

    def encode(self, s: str) -> [int]:
        """Encodes a string into a list of token ids.

        Args:
            s (str): _description_

        Returns:
            [int]: _description_
        """
        return self.sp_model.encode(s, out_type=int)

    def decode(self, ids: [int]) -> str:
        """decodes a list of token ids into a string.

        Args:
            ids (int]): _description_

        Returns:
            str: _description_
        """
        return self.sp_model.decode(ids)
