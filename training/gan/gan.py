import logging
import os
import shutil
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from datasets import load_dataset
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Generator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    num_epochs: int,
    latent_dim: int,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(
        generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )
    d_optimizer = optim.Adam(
        discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
    )

    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            real_samples = batch["audio"].to(device)
            batch_size = real_samples.size(0)

            d_optimizer.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            d_real_output = discriminator(real_samples)
            d_real_loss = criterion(d_real_output, real_labels)

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_samples = generator(z)
            d_fake_output = discriminator(fake_samples.detach())
            d_fake_loss = criterion(d_fake_output, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_samples = generator(z)
            d_output = discriminator(fake_samples)
            g_loss = criterion(d_output, real_labels)
            g_loss.backward()
            g_optimizer.step()

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}"
        )

    return generator, discriminator


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Clear the dataset cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info("Cleared dataset cache")

    # Try to load the dataset with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            dataset = load_dataset(
                "mozilla-foundation/common_voice_11_0",
                "en",
                split="train[:1000]",
                trust_remote_code=True,
            )
            logger.info(f"Dataset loaded: {len(dataset)} samples")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Dataset loading failed (attempt {attempt + 1}/{max_retries}). Retrying..."
                )
            else:
                logger.error("Failed to load dataset after multiple attempts.")
                raise e

    def preprocess_audio(example):
        audio = example["audio"]["array"]
        resampled_audio = torchaudio.transforms.Resample(
            example["audio"]["sampling_rate"], 16000
        )(torch.tensor(audio))
        return {"audio": resampled_audio.flatten()}

    dataset = dataset.map(preprocess_audio, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["audio"])

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    latent_dim = 100
    output_dim = 16000

    generator = Generator(latent_dim, output_dim).to(device)
    discriminator = Discriminator(output_dim).to(device)

    num_epochs = 50
    generator, discriminator = train_gan(
        generator, discriminator, dataloader, num_epochs, latent_dim, device
    )

    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    logger.info("Models saved successfully")


if __name__ == "__main__":
    main()
