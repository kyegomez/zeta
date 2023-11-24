import logging
import torch
from transformers import CLIPProcessor, AutoTokenizer

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MultiModalTokenizer:
    """
    A tokenizer class for the kosmos model

    Attributes:
        processor(CLIPProcessor): The processor to tokenize images
        tokenizer: (AutoTokenizer): The tokenizer to tokenize text
        im_idx: (int): The Index of the "<image>" token.
        im_end_idx (int): The index of the "</image>" token.
    """

    def __init__(self, max_length: int = 8192):
        self.max_length = max_length

        try:
            self.processor = CLIPProcessor.from_pretrained(
                "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
                additional_special_tokens=["<image>", "</image>"],
                eos_token="<eos>",
                pad_token="<pad>",
                extra_ids=0,
                model_max_length=self.max_length,
            )
        except Exception as e:
            logging.error(f"Failed to initialize KosmosTokenizer: {e}")
            raise

        self.im_idx, self.im_end_idx = self.tokenizer.convert_tokens_to_ids(
            ["<image>", "</image>"]
        )

    def tokenize_texts(self, texts: str):
        """
        Tokenize given texts.

        Args:
            Texts (str): The Text to be tokenized


        Returns:
            A tuple containing the tokenized texts and only the text tokens.
        """
        try:
            texts = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            ).input_ids
            # Add image tokens to text as "<s> <image> </image> text </s>"
            image_tokens = torch.tensor(
                [[self.im_idx, self.im_end_idx]] * texts.shape[0]
            )
            return (
                torch.cat([texts[:, 0:1], image_tokens, texts[:, 1:]], dim=1),
                texts,
            )
        except Exception as e:
            logging.error(f"Failed to tokenize texts: {e}")
            raise

    def tokenize_images(self, images):
        """
        Tokenizes given images.

        Args:
            images: The images to be tokenized

        Returns:
            The tokenized images.

        """
        try:
            return self.processor(
                images=images, return_tensors="pt"
            ).pixel_values
        except Exception as e:
            logging.error(f"Failed to tokenize images: {e}")
            raise

    def tokenize(self, sample):
        """
        Tokenizes given sample.

        Args:
            Sample: The sample to be tokenized

        Returns:
            A dictionary containing the tokenized text tokens, images, labels, and attention mask.

        """
        try:
            text_tokens, only_text_tokens = self.tokenize_texts(
                sample["target_text"]
            )
            attention_mask = text_tokens != self.tokenizer.pad_token_id
            dummy_image_features = torch.ones((text_tokens.shape[0], 64))
            attention_mask = torch.cat(
                [dummy_image_features, attention_mask], dim=1
            )
            return {
                "text_tokens": text_tokens,
                "images": self.tokenize_images(sample["image"]),
                "labels": only_text_tokens,
                "attention_mask": attention_mask,
            }
        except Exception as e:
            logging.error(f"Failed to tokenize sample: {e}")
            raise
