from transformers import AutoTokenizer


class LanguageTokenizerGPTX:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b",
            eos_token="<eos>",
            pad_token="<pad>",
            extra_ids=0,
            model_max_length=8192,
        )

    def tokenize_texts(self, texts):
        return self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).input_ids

    def decode(self, texts):
        return self.tokenizer.decode(texts)

    def __len__(self):
        num_tokens = len(self.tokenizer)
        return num_tokens
