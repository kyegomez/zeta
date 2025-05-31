import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW
from typing import List, Tuple, Dict
from datasets import load_dataset
from loguru import logger
from zeta.utils import verbose_execution
import numpy as np

# Set up loguru logger
logger.add("rf_prediction.log", rotation="10 MB")


class RFDataset(Dataset):
    def __init__(self, data: List[Dict[str, torch.Tensor]]):
        """
        Initialize the RF dataset.

        Args:
            data (List[Dict[str, torch.Tensor]]): List of dictionaries containing RF data.
        """
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


@verbose_execution(log_params=True, log_gradients=True, log_memory=True)
class RFTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the RF Transformer model.

        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output predictions.
        """
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.audio_encoder = nn.Linear(input_dim, self.bert.config.hidden_size)

    def forward(
        self, input_ids: torch.Tensor, audio_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the RF Transformer model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
            audio_input (torch.Tensor): Audio input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output predictions.
        """
        bert_output = self.bert(input_ids).last_hidden_state[:, 0, :]
        audio_encoded = self.audio_encoder(audio_input)
        combined_input = bert_output + audio_encoded
        return self.fc(combined_input)


def load_rf_data(dataset_name: str) -> Tuple[RFDataset, RFDataset]:
    """
    Load RF data from Hugging Face datasets.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face.

    Returns:
        Tuple[RFDataset, RFDataset]: Train and validation datasets.
    """
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    # Get available splits
    splits = list(dataset.keys())
    logger.info(f"Available splits: {splits}")

    if len(splits) == 0:
        raise ValueError(f"Dataset {dataset_name} has no splits.")

    # Use the first (and possibly only) split for both train and validation
    split = splits[0]
    logger.info(f"Using split '{split}' for both training and validation")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    max_length = 512  # BERT's maximum sequence length

    def process_item(item):
        audio = item["audio"]["array"][:max_length]  # Truncate to max_length
        padded_audio = np.pad(
            audio, (0, max_length - len(audio))
        )  # Pad if necessary
        input_ids = tokenizer.encode(
            item["text"],
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        return {
            "input": torch.tensor(padded_audio, dtype=torch.float),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor([float(item["text"] != "")]),
        }

    all_data = [process_item(d) for d in dataset[split]]

    # Shuffle the data
    import random

    random.shuffle(all_data)

    # Split the data into train (80%) and validation (20%)
    split_point = int(len(all_data) * 0.8)
    train_data = all_data[:split_point]
    val_data = all_data[split_point:]

    logger.info(
        f"Dataset loaded. Train size: {len(train_data)}, Validation size: {len(val_data)}"
    )
    return RFDataset(train_data), RFDataset(val_data)


def train_rf_model(
    model: RFTransformer,
    train_dataset: RFDataset,
    val_dataset: RFDataset,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
) -> RFTransformer:
    """
    Train the RF Transformer model.

    Args:
        model (RFTransformer): The RF Transformer model to train.
        train_dataset (RFDataset): Training dataset.
        val_dataset (RFDataset): Validation dataset.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimization.

    Returns:
        RFTransformer: Trained RF Transformer model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, audio_inputs, labels = (
                batch["input_ids"].to(device),
                batch["input"].to(device),
                batch["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(input_ids, audio_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, audio_inputs, labels = (
                    batch["input_ids"].to(device),
                    batch["input"].to(device),
                    batch["label"].to(device),
                )
                outputs = model(input_ids, audio_inputs)
                val_loss += criterion(outputs, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
        )

    return model


def save_model(model: RFTransformer, path: str) -> None:
    """
    Save the trained RF Transformer model.

    Args:
        model (RFTransformer): Trained RF Transformer model.
        path (str): Path to save the model.
    """
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


def load_model(path: str, input_dim: int, output_dim: int) -> RFTransformer:
    """
    Load a trained RF Transformer model.

    Args:
        path (str): Path to the saved model.
        input_dim (int): Dimension of input features.
        output_dim (int): Dimension of output predictions.

    Returns:
        RFTransformer: Loaded RF Transformer model.
    """
    model = RFTransformer(input_dim, output_dim)
    model.load_state_dict(torch.load(path))
    logger.info(f"Model loaded from {path}")
    return model


def predict(
    model: RFTransformer, input_ids: torch.Tensor, audio_input: torch.Tensor
) -> torch.Tensor:
    """
    Make predictions using the trained RF Transformer model.

    Args:
        model (RFTransformer): Trained RF Transformer model.
        input_ids (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
        audio_input (torch.Tensor): Audio input tensor of shape (batch_size, input_dim).

    Returns:
        torch.Tensor: Predicted output.
    """
    model.eval()
    with torch.no_grad():
        return model(input_ids, audio_input)


if __name__ == "__main__":
    logger.info("Starting RF prediction script")

    # Example usage
    dataset_name = "hf-internal-testing/librispeech_asr_dummy"
    train_dataset, val_dataset = load_rf_data(dataset_name)

    input_dim = train_dataset[0]["input"].shape[0]
    output_dim = train_dataset[0]["label"].shape[0]

    logger.info(f"Input dimension: {input_dim}, Output dimension: {output_dim}")

    model = RFTransformer(input_dim, output_dim)
    trained_model = train_rf_model(
        model,
        train_dataset,
        val_dataset,
        num_epochs=5,
        batch_size=32,
        learning_rate=1e-4,
    )

    model_path = "rf_transformer_model.pth"
    save_model(trained_model, model_path)

    # Load the model and make predictions
    loaded_model = load_model(model_path, input_dim, output_dim)
    sample_input = torch.randn(1, input_dim)
    sample_input_ids = torch.randint(
        0, 1000, (1, 512)
    )  # Random input_ids for demonstration
    prediction = predict(loaded_model, sample_input_ids, sample_input)
    logger.info(f"Prediction: {prediction.item():.4f}")

    logger.info("RF prediction script completed")
