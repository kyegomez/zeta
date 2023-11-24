from itertools import chain
from datasets import load_dataset
from transformers import AutoTokenizer


def build_dataloaders(seq_len: int = None, num_cpu: int = None):
    """
    Build data loaders for training.

    This function performs the following steps:
    1. Load the tokenizer from the pretrained "EleutherAI/gpt-neox-20b" model.
    2. Load the "openwebtext" dataset.
    3. Tokenize the dataset, adding the end-of-sentence token to each text.
    4. Process the tokenized dataset into chunks of a specified block size.

    Returns:
        Dataset: The processed dataset ready for training.
    """
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    dataset = load_dataset("openwebtext", split="train")

    tokenized_dataset = dataset.map(
        lambda example: tokenizer(
            [t + tokenizer.eos_token for t in example["text"]]
        ),
        batched=True,
        num_proc=seq_len,
        remove_columns=["text"],
    )

    block_size = seq_len

    # Main data processing function that will concatenate all texts from our
    # dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    train_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=num_cpu,
    )

    return train_dataset


def build_pre_tokenized(dataset_name: str = None):
    d0 = load_dataset(dataset_name)
    # d1 = load_dataset("conceptofmind/c4_21-to-40_neox_with_eos_8k", split="train")
    # d2 = load_dataset("conceptofmind/c4_41-to-60_neox_with_eos_8k", split="train")
    # d3 = load_dataset("conceptofmind/c4_61-to-80_neox_with_eos_8k", split="train")
    # d4 = load_dataset("conceptofmind/c4_81-to-100_neox_with_eos_8k", split="train")
    # train_dataset = concatenate_datasets([d0, d1, d2, d3, d4])
    return d0
