# WIP
import os

import kaggle
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Set up logging
logger.add("weather_prediction.log", rotation="10 MB", level="INFO")

# Constants and configuration
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 50
LEARNING_RATE: float = 1e-4
IMAGE_SIZE: int = 224
DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
DATASET_NAME: str = "pratik2901/multiclass-weather-dataset"
DATASET_PATH: str = "weather_data"


def download_and_extract_dataset():
    """Download and extract the dataset from Kaggle."""
    logger.info("Downloading dataset from Kaggle")
    kaggle.api.dataset_download_files(
        DATASET_NAME, path=DATASET_PATH, unzip=True
    )


def load_and_process_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load and preprocess the Weather Image Dataset.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Preprocessed dataframe and array of class names.
    """
    logger.info("Loading and preprocessing dataset")

    if not os.path.exists(DATASET_PATH):
        download_and_extract_dataset()

    data = []
    for class_name in os.listdir(DATASET_PATH):
        class_path = os.path.join(DATASET_PATH, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                data.append({"image_path": img_path, "weather": class_name})

    df = pd.DataFrame(data)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["weather"])

    logger.info(
        f"Dataset loaded with {len(df)} samples and {len(le.classes_)} classes"
    )
    return df, le.classes_


class WeatherDataset(Dataset):
    """Custom dataset for weather images."""

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.dataframe.iloc[idx]["image_path"]
        label = self.dataframe.iloc[idx]["label"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class WeatherViT(nn.Module):
    """Vision Transformer model for weather prediction."""

    def __init__(self, num_classes: int):
        super(WeatherViT, self).__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
) -> nn.Module:
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total
        val_acc = evaluate(model, val_loader)

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_weather_vit.pth")
            logger.info(
                f"Best model saved with validation accuracy: {best_val_acc:.2f}%"
            )

    return model


def evaluate(model: nn.Module, data_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


@logger.catch
def main():
    """Main function to run the weather prediction model training and evaluation."""
    logger.info("Starting weather prediction model training")

    # Load dataset
    df, classes = load_and_process_dataset()
    logger.info(
        f"Dataset loaded with {len(df)} samples and {len(classes)} classes"
    )

    # Split dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=42
    )

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # Create datasets and data loaders
    train_dataset = WeatherDataset(train_df, transform)
    val_dataset = WeatherDataset(val_df, transform)
    test_dataset = WeatherDataset(test_df, transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Initialize model, loss function, and optimizer
    model = WeatherViT(len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    logger.info("Starting model training")
    model = train(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS
    )

    # Evaluate on test set
    test_acc = evaluate(model, test_loader)
    logger.info(f"Final test accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
