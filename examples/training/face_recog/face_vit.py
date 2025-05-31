import os
from typing import Tuple, List
import urllib.request
import tarfile
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from timm import create_model
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

# Constants
DATASET_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
DATASET_PATH = "lfw.tgz"
EXTRACTED_PATH = "lfw"
MIN_SAMPLES_PER_CLASS = 2


class FacialRecognitionDataset(Dataset):
    def __init__(
        self, image_paths: List[str], labels: List[int], transform=None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def download_and_extract_dataset():
    if not os.path.exists(DATASET_PATH):
        print("Downloading LFW dataset...")
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)

    if not os.path.exists(EXTRACTED_PATH):
        print("Extracting dataset...")
        with tarfile.open(DATASET_PATH, "r:gz") as tar:
            tar.extractall()


def load_dataset(data_dir: str) -> Tuple[List[str], List[int]]:
    image_paths = []
    labels = []
    label_to_index = {}

    for person_name in tqdm(os.listdir(data_dir), desc="Loading dataset"):
        person_dir = os.path.join(data_dir, person_name)
        if os.path.isdir(person_dir):
            person_images = [
                os.path.join(person_dir, img)
                for img in os.listdir(person_dir)
                if img.endswith((".jpg", ".jpeg", ".png"))
            ]
            if len(person_images) >= MIN_SAMPLES_PER_CLASS:
                if person_name not in label_to_index:
                    label_to_index[person_name] = len(label_to_index)
                image_paths.extend(person_images)
                labels.extend(
                    [label_to_index[person_name]] * len(person_images)
                )

    return image_paths, labels


def create_data_loaders(
    image_paths: List[str], labels: List[int], batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Count samples per class
    class_sample_count = Counter(labels)

    # Filter out classes with less than 3 samples (to ensure at least 2 for train and 1 for test)
    valid_samples = [
        (path, label)
        for path, label in zip(image_paths, labels)
        if class_sample_count[label] >= 3
    ]

    if len(valid_samples) < len(labels):
        print(
            f"Removed {len(labels) - len(valid_samples)} samples from classes with less than 3 members."
        )

    valid_image_paths, valid_labels = zip(*valid_samples)

    # Double-check that we have at least three samples per class
    class_sample_count = Counter(valid_labels)
    if min(class_sample_count.values()) < 3:
        raise ValueError(
            "After filtering, there are still classes with less than 3 samples."
        )

    print(f"Valid images after filtering: {len(valid_image_paths)}")
    print(f"Valid classes after filtering: {len(set(valid_labels))}")

    # Print class distribution
    print("Class distribution:")
    for label, count in sorted(class_sample_count.items()):
        print(f"Class {label}: {count} samples")

    dataset = FacialRecognitionDataset(
        list(valid_image_paths),
        list(valid_labels),
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    # Try stratified split, fall back to random split if it fails
    try:
        train_idx, test_idx = train_test_split(
            range(len(valid_labels)),
            test_size=0.2,
            stratify=valid_labels,
            random_state=42,
        )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=0.1,
            stratify=[valid_labels[i] for i in train_idx],
            random_state=42,
        )
    except ValueError as e:
        print(f"Stratified split failed: {e}")
        print("Falling back to random split.")
        train_idx, test_idx = train_test_split(
            range(len(valid_labels)), test_size=0.2, random_state=42
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.1, random_state=42
        )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = train_correct / train_total
        val_acc = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(
            f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}"
        )
        print(f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

        scheduler.step()

    return model


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_and_extract_dataset()

    image_paths, labels = load_dataset(EXTRACTED_PATH)

    print(f"Total images: {len(image_paths)}")
    print(f"Total classes: {len(set(labels))}")

    train_loader, val_loader, test_loader = create_data_loaders(
        image_paths, labels, batch_size=32
    )

    num_classes = len(set(labels))
    model = create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=num_classes
    )
    model = model.to(device)

    num_epochs = 20
    model = train_model(model, train_loader, val_loader, num_epochs, device)

    test_acc = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
