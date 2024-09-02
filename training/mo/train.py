# training/train_mamba.py
import argparse
import json
import os
from multiprocessing import Pool
from typing import Any, Dict, List

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import load_dataset
from loguru import logger
from mamba_omega import MambaConfig, MambaLMHeadModel
from tiktoken import encoding_for_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def count_available_gpus():
    """
    Count the number of available GPUs.

    Returns:
        int: Number of available GPUs.
    """
    return torch.cuda.device_count()


def setup_distributed(rank, world_size):
    """
    Set up distributed training.

    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


# Configuration management
def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


# Tokenizer
TOKENIZER_MODEL = "o200k_base"
tokenizer = encoding_for_model(TOKENIZER_MODEL)


def load_and_preprocess_dataset(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Load and preprocess the dataset using multiprocessing.

    Returns:
        List[Dict[str, Any]]: Tokenized dataset.
    """
    logger.info("Loading dataset...")
    dataset = load_dataset(
        config["dataset"]["name"],
        name=config["dataset"]["split"],
        split="train",
        streaming=True,
    )

    def preprocess_data(data: Dict[str, Any]) -> List[int]:
        """
        Preprocess a single data item by merging URL and text, then tokenizing.

        Args:
            data (Dict[str, Any]): A single data item.

        Returns:
            List[int]: Tokenized data.
        """
        text = data["url"] + " " + data["text"]
        return tokenizer.encode(text)

    logger.info("Preprocessing dataset...")
    with Pool() as pool:
        tokenized_data = pool.map(preprocess_data, dataset)

    return tokenized_data


class FineWebDataset(Dataset):
    def __init__(self, data: List[List[int]]):
        """
        Initialize the dataset with tokenized data.

        Args:
            data (List[List[int]]): Tokenized data.
        """
        self.data = data

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            torch.Tensor: Tokenized data as a tensor.
        """
        return torch.tensor(self.data[idx])


def create_dataloader(
    data: List[List[int]],
    batch_size: int = 32,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Create a DataLoader for the dataset.

    Args:
        data (List[List[int]]): Tokenized data.
        batch_size (int, optional): Batch size. Defaults to 32.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    dataset = FineWebDataset(data)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def initialize_model(
    config: Dict[str, Any], device: torch.device
) -> MambaLMHeadModel:
    """
    Initialize the Mamba model with the specified configuration.

    Returns:
        MambaLMHeadModel: Initialized Mamba model.
    """
    logger.info("Initializing model with configuration: {}", config)
    model_config = MambaConfig(
        d_model=config["model"]["d_model"],
        n_layer=config["model"]["n_layer"],
        d_intermediate=config["model"]["d_intermediate"],
        vocab_size=tokenizer.n_vocab,
        ssm_cfg=config["model"]["ssm_cfg"],
        attn_layer_idx=config["model"]["attn_layer_idx"],
        attn_cfg=config["model"]["attn_cfg"],
        rms_norm=config["model"]["rms_norm"],
        residual_in_fp32=config["model"]["residual_in_fp32"],
        fused_add_norm=config["model"]["fused_add_norm"],
        pad_vocab_size_multiple=config["model"]["pad_vocab_size_multiple"],
        tie_embeddings=config["model"]["tie_embeddings"],
    )
    model = MambaLMHeadModel(model_config)
    model = model.to(device)
    model = DDP(model, device_ids=[device])
    logger.info("Model initialized and moved to device: {}", device)
    return model


def save_checkpoint(model_engine, epoch: int, output_dir: str) -> None:
    """
    Save model checkpoint.

    Args:
        model_engine: The model engine.
        epoch (int): Current epoch.
        output_dir (str): Directory to save the checkpoint.
    """
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
    model_engine.save_checkpoint(output_dir, f"checkpoint_epoch_{epoch}")
    logger.info(f"Checkpoint saved at {checkpoint_path}")
    logger.info("Loguru now")


def train_model(
    model: MambaLMHeadModel,
    dataloader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    rank: int,
) -> None:
    """
    Train the Mamba model.

    Args:
        model (MambaLMHeadModel): The Mamba model to train.
        dataloader (DataLoader): DataLoader for the training data.
        config (Dict[str, Any]): Configuration dictionary.
        device (torch.device): Device to train on.
        rank (int): Rank of the current process.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss()

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=config["deepspeed_config"]
    )

    logger.info("Starting training...")
    for epoch in range(config["training"]["epochs"]):
        dataloader.sampler.set_epoch(epoch)
        model_engine.train()
        for batch in dataloader:
            inputs = batch.to(device)
            labels = inputs.clone()
            outputs = model_engine(inputs).logits
            loss = criterion(
                outputs.view(-1, outputs.size(-1)), labels.view(-1)
            )

            model_engine.backward(loss)
            model_engine.step()

            if rank == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item()}")

        # Save checkpoint
        if rank == 0:
            save_checkpoint(model_engine, epoch, config["output_dir"])

    logger.info("Training complete.")


def main(config_path: str, rank: int, world_size: int) -> None:
    """
    Main function to run the training script.

    Args:
        config_path (str): Path to the configuration file.
        rank (int): Rank of the current process.
        world_size (int): Total number of processes.
    """
    # Load configuration
    config = load_config(config_path)

    # Set up distributed training
    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Load and preprocess dataset
    tokenized_data = load_and_preprocess_dataset(config)

    # Create DataLoader
    train_loader = create_dataloader(
        tokenized_data, config["training"]["batch_size"], rank, world_size
    )

    # Initialize model
    model = initialize_model(config, device)

    # Train model
    train_model(model, train_loader, config, device, rank)

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Mamba model on FineWeb dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    world_size = count_available_gpus()
    torch.multiprocessing.spawn(
        main, args=(args.config, world_size), nprocs=world_size, join=True
    )
