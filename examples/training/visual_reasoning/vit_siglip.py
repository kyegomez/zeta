import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from timm import create_model
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset


class HuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["img"]
        label = item["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders(
    batch_size: int, num_workers: int
) -> Dict[str, DataLoader]:
    """
    Create training and validation dataloaders using a Hugging Face dataset.

    Args:
        batch_size (int): Batch size for training and validation.
        num_workers (int): Number of data loading workers.

    Returns:
        Dict[str, DataLoader]: Dictionary containing train and val dataloaders.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]
    )

    # Load CIFAR-100 dataset from Hugging Face
    dataset = load_dataset("cifar100")

    train_dataset = HuggingFaceDataset(dataset["train"], transform=transform)
    val_dataset = HuggingFaceDataset(dataset["test"], transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return {"train": train_loader, "val": val_loader}


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): The training dataloader.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        device (torch.device): The device to use for training.

    Returns:
        Tuple[float, float]: Average loss and accuracy for the epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Validate the model.

    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): The validation dataloader.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to use for validation.

    Returns:
        Tuple[float, float]: Average loss and accuracy for the validation set.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def train_vit(
    model_name: str,
    num_classes: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    num_workers: int,
    device: str,
    output_dir: str,
):
    """
    Train a Vision Transformer model.

    Args:
        model_name (str): Name of the ViT model from timm.
        num_classes (int): Number of classes in the dataset.
        batch_size (int): Batch size for training and validation.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Initial learning rate.
        weight_decay (float): Weight decay for optimizer.
        num_workers (int): Number of data loading workers.
        device (str): Device to use for training ('cuda' or 'cpu').
        output_dir (str): Directory to save the trained model.
    """
    device = torch.device(device)

    # Create dataloaders
    dataloaders = create_dataloaders(batch_size, num_workers)

    # Create model
    model = create_model(model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, dataloaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, dataloaders["val"], criterion, device
        )

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save the trained model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        model.state_dict(), os.path.join(output_dir, f"{model_name}_final.pth")
    )


if __name__ == "__main__":
    # Set your training parameters here
    train_vit(
        model_name="vit_base_patch16_224",
        num_classes=100,  # CIFAR-100 has 100 classes
        batch_size=64,
        num_epochs=50,
        learning_rate=1e-4,
        weight_decay=1e-4,
        num_workers=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir="output",
    )
