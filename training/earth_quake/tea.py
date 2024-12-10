import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
import requests
from datetime import timedelta
from loguru import logger
from datetime import datetime

# Set up loguru logger
logger.add("earthquake_prediction.log", rotation="10 MB", level="DEBUG")

# Constants
SEQUENCE_LENGTH = 30  # Number of time steps to consider
FEATURE_DIM = 5  # Number of features per time step (time, latitude, longitude, depth, magnitude)
NUM_CLASSES = 2  # Binary classification: earthquake or no earthquake
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4


# @verbose_execution(log_params=True, log_gradients=True, log_memory=True)
class EarthquakeTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super(EarthquakeTransformer, self).__init__()
        self.embedding = nn.Linear(feature_dim, 64)
        self.positional_encoding = self.create_positional_encoding(
            SEQUENCE_LENGTH, 64
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(64, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x + self.positional_encoding.to(x.device)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

    @staticmethod
    def create_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
        )
        pos_encoding = torch.zeros(seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding


class EarthquakeDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def fetch_earthquake_data(
    start_date: str, end_date: str, min_magnitude: float = 2.5
) -> pd.DataFrame:
    """
    Fetch earthquake data from the USGS Earthquake Catalog API in smaller time chunks.
    """
    logger.info(f"Fetching earthquake data from {start_date} to {end_date}")
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_earthquakes = []
    chunk_size = timedelta(days=30)  # Fetch data in 30-day chunks

    current_start = start
    while current_start < end:
        current_end = min(current_start + chunk_size, end)
        params = {
            "format": "geojson",
            "starttime": current_start.strftime("%Y-%m-%d"),
            "endtime": current_end.strftime("%Y-%m-%d"),
            "minmagnitude": min_magnitude,
            "orderby": "time",
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            chunk_earthquakes = []
            for feature in data["features"]:
                props = feature["properties"]
                coords = feature["geometry"]["coordinates"]
                chunk_earthquakes.append(
                    {
                        "time": props["time"],
                        "latitude": coords[1],
                        "longitude": coords[0],
                        "depth": coords[2],
                        "magnitude": props["mag"],
                    }
                )
            all_earthquakes.extend(chunk_earthquakes)
            logger.info(
                f"Fetched {len(chunk_earthquakes)} earthquakes from {current_start.date()} to {current_end.date()}"
            )

        except requests.exceptions.RequestException as e:
            logger.error(
                f"Error fetching earthquake data for {current_start.date()} to {current_end.date()}: {e}"
            )

        current_start = current_end + timedelta(days=1)

    logger.info(f"Fetched a total of {len(all_earthquakes)} earthquake records")
    return pd.DataFrame(all_earthquakes)


def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the earthquake data and create sequences for training.
    """
    logger.info("Preprocessing earthquake data")
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.sort_values("time")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(
        df[["latitude", "longitude", "depth", "magnitude"]]
    )

    sequences = []
    labels = []

    for i in range(len(df) - SEQUENCE_LENGTH):
        seq = scaled_data[i : i + SEQUENCE_LENGTH]
        sequences.append(seq)

        # Label is 1 if there's an earthquake with magnitude >= 5.0 in the next 24 hours
        next_day = df.iloc[i + SEQUENCE_LENGTH]["time"] + timedelta(days=1)
        future_quakes = df[
            (df["time"] > df.iloc[i + SEQUENCE_LENGTH]["time"])
            & (df["time"] <= next_day)
        ]
        label = 1 if (future_quakes["magnitude"] >= 5.0).any() else 0
        labels.append(label)

    logger.info(f"Created {len(sequences)} sequences for training")
    return np.array(sequences), np.array(labels)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
) -> Dict[str, List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    model.to(device)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / len(train_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )

    return history


def main():
    logger.info("Starting earthquake prediction model training")

    # Fetch and preprocess data
    start_date = "2010-01-01"
    end_date = "2023-04-14"
    df = fetch_earthquake_data(start_date, end_date)
    sequences, labels = preprocess_data(df)

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    logger.info(
        f"Train set size: {len(X_train)}, Validation set size: {len(X_val)}"
    )

    # Create datasets and dataloaders
    train_dataset = EarthquakeDataset(X_train, y_train)
    val_dataset = EarthquakeDataset(X_val, y_val)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model, loss function, and optimizer
    model = EarthquakeTransformer(
        feature_dim=FEATURE_DIM, num_heads=4, num_layers=2
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logger.info("Starting model training")
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

    # Save the trained model
    torch.save(model.state_dict(), "earthquake_transformer.pth")
    logger.info(
        "Training completed. Model saved as 'earthquake_transformer.pth'"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")
