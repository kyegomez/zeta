import torch
from transformers import T5Tokenizer, T5EncoderModel
from loguru import logger


class PretrainedT5Embedder:
    def __init__(self, model_name: str = "t5-small", *args, **kwargs):
        """
        Initializes the PretrainedT5Embedder with a specified T5 model.

        Args:
            model_name (str): The name of the pre-trained T5 model to use.
        """
        logger.info(
            f"Initializing the T5 tokenizer and model with {model_name}."
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name, *args, **kwargs)

    def run(self, text: str, *args, **kwargs) -> torch.Tensor:
        """
        Encodes the input text using the T5 model and returns the embeddings.

        Args:
            text (str): The input text to be embedded.

        Returns:
            torch.Tensor: The embedded representation of the input text.
        """
        logger.info(f"Encoding the text: {text}")
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        logger.info("Text successfully embedded.")
        return embeddings
