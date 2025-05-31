from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import LFWPeople

# Set up logging
logger.add("facial_recognition_training.log", rotation="10 MB")


class FacialRecognitionCNN(nn.Module):
    """Convolutional Neural Network for facial recognition."""

    def __init__(self, num_classes: int):
        """
        Initialize the CNN model.

        Args:
            num_classes (int): Number of output classes.
        """
        super(FacialRecognitionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_data_transforms() -> transforms.Compose:
    """
    Get data transformations for preprocessing.

    Returns:
        transforms.Compose: Composition of transforms.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


class RemappedDataset(Dataset):
    def __init__(self, dataset, remapped_targets):
        self.dataset = dataset
        self.remapped_targets = remapped_targets

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        return img, self.remapped_targets[index]

    def __len__(self):
        return len(self.dataset)


def load_dataset(data_dir: str) -> Tuple[Dataset, Dataset, int]:
    """
    Load and prepare the LFW dataset.

    Args:
        data_dir (str): Directory to store the dataset.

    Returns:
        Tuple[Dataset, Dataset, int]: Train dataset, validation dataset, and number of classes.
    """
    transform = get_data_transforms()
    full_dataset = LFWPeople(
        root=data_dir, split="train", download=True, transform=transform
    )

    logger.info(f"Full dataset size: {len(full_dataset)}")
    logger.info(f"Original number of classes: {len(set(full_dataset.targets))}")

    # Remap labels to a continuous range
    label_encoder = LabelEncoder()
    remapped_targets = label_encoder.fit_transform(full_dataset.targets)
    num_classes = len(label_encoder.classes_)

    logger.info(f"Remapped number of classes: {num_classes}")
    logger.info(
        f"Target distribution: {torch.bincount(torch.tensor(remapped_targets))}"
    )

    remapped_dataset = RemappedDataset(full_dataset, remapped_targets)

    try:
        train_indices, val_indices = train_test_split(
            range(len(remapped_dataset)),
            test_size=0.2,
            stratify=remapped_targets,
            random_state=42,
        )
    except ValueError as e:
        logger.warning(
            f"Stratified split failed: {e}. Falling back to random split."
        )
        train_indices, val_indices = train_test_split(
            range(len(remapped_dataset)),
            test_size=0.2,
            stratify=None,
            random_state=42,
        )

    train_dataset = torch.utils.data.Subset(remapped_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(remapped_dataset, val_indices)

    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")

    return train_dataset, val_dataset, num_classes


def train(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / total
        progress_bar.set_postfix(
            {
                "loss": f"{running_loss/len(progress_bar):.4f}",
                "accuracy": f"{accuracy:.2f}%",
            }
        )
    return running_loss / len(dataloader), accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc="Validating")
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            accuracy = 100.0 * correct / total
            progress_bar.set_postfix(
                {
                    "loss": f"{running_loss/len(progress_bar):.4f}",
                    "accuracy": f"{accuracy:.2f}%",
                }
            )
    return running_loss / len(dataloader), accuracy


def main():
    logger.info("Starting facial recognition model training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    batch_size = 16
    num_epochs = 39
    learning_rate = 0.0001

    data_dir = "data/lfw"
    train_dataset, val_dataset, num_classes = load_dataset(data_dir)

    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = FacialRecognitionCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=3, factor=0.1
    )

    logger.info("Starting training")
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_accuracy = train(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        logger.info(
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%"
        )
        logger.info(
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )

        scheduler.step(val_loss)

    torch.save(model.state_dict(), "facial_recognition_model_lfw.pth")
    logger.info("Training completed. Model saved.")


if __name__ == "__main__":
    main()
