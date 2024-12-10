# Food101
# classtorchvision.datasets.Food101(root: Union[str, Path], split: str = 'train', transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)[source]
# The Food-101 Data Set.

# The Food-101 is a challenging data set of 101 food categories with 101,000 images. For each class, 250 manually reviewed test images are provided as well as 750 training images. On purpose, the training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels.

# Parameters:
# root (str or pathlib.Path) – Root directory of the dataset.

# split (string, optional) – The dataset split, supports "train" (default) and "test".

# transform (callable, optional) – A function/transform that takes in a PIL image and returns a transformed version. E.g, transforms.RandomCrop.

# target_transform (callable, optional) – A function/transform that takes in the target and transforms it.

# download (bool, optional) – If True, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. Default is False.

# Special-members:
# __getitem__(idx: int) → Tuple[Any, Any][source]
# Parameters:
# index (int) – Index

# Returns:
# Sample and meta data, optionally transformed by the respective transforms.

# Return type:
# (Any)
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.quantization import quantize_dynamic
from torchvision.models import vit_b_16, ViT_B_16_Weights
from typing import Union, Tuple
from pathlib import Path
from loguru import logger
import time


def get_data_loaders(
    root: Union[str, Path], batch_size: int = 32, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Get the training and test data loaders for the Food101 dataset.

    Args:
        root (Union[str, Path]): Root directory of the dataset.
        batch_size (int): Batch size for the data loaders.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and test data loaders.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    train_dataset = datasets.Food101(
        root=root, split="train", transform=transform, download=True
    )
    test_dataset = datasets.Food101(
        root=root, split="test", transform=transform, download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def finetune_vit(
    train_loader: DataLoader, num_classes: int = 101, epochs: int = 10
) -> nn.Module:
    """
    Fine-tune a pretrained Vision Transformer (ViT) model on the Food101 dataset.

    Args:
        train_loader (DataLoader): Training data loader.
        num_classes (int): Number of classes in the dataset.
        epochs (int): Number of epochs to train the model.

    Returns:
        nn.Module: Fine-tuned ViT model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_end_time = time.time()
            logger.info(
                f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Batch Time: {batch_end_time - batch_start_time:.4f}s, Loss: {loss.item():.4f}"
            )
        epoch_end_time = time.time()
        logger.info(
            f"Epoch [{epoch+1}/{epochs}], Epoch Time: {epoch_end_time - epoch_start_time:.4f}s, Average Loss: {running_loss/len(train_loader):.4f}"
        )

    return model


def quantize_model(model: nn.Module) -> nn.Module:
    """
    Quantize the given model to make it faster.

    Args:
        model (nn.Module): The model to be quantized.

    Returns:
        nn.Module: Quantized model.
    """
    model.eval()
    quantized_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return quantized_model


def main():
    """
    Main function to fine-tune and quantize the ViT model on the Food101 dataset.
    """
    logger.add("training.log", rotation="500 MB")
    logger.info("Starting the training and quantization process.")

    root = "./data"
    batch_size = 32
    epochs = 10
    num_workers = 4

    train_loader, test_loader = get_data_loaders(root, batch_size, num_workers)
    model = finetune_vit(train_loader, epochs=epochs)
    quantized_model = quantize_model(model)

    torch.save(quantized_model.state_dict(), "quantized_vit_food101.pth")
    logger.info(
        "Model training and quantization complete. Model saved as 'quantized_vit_food101.pth'."
    )


if __name__ == "__main__":
    main()
