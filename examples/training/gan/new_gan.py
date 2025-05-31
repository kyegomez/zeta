import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3, dilation=dilation, padding="same"
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3, dilation=dilation, padding="same"
        )
        self.norm1 = nn.GroupNorm(
            num_groups=channels // 4, num_channels=channels
        )
        self.norm2 = nn.GroupNorm(
            num_groups=channels // 4, num_channels=channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.norm2(self.conv2(x)), 0.1)
        return x + residual


class Generator(nn.Module):
    """Dynamic generator for speech synthesis."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int = 512,
        upsample_rates: List[int] = [8, 8, 2, 2],
    ):
        super().__init__()
        self.input_conv = nn.Conv1d(
            input_channels, base_channels, kernel_size=7, padding=3
        )
        self.norm = nn.GroupNorm(
            num_groups=base_channels // 4, num_channels=base_channels
        )

        self.upsample_layers = nn.ModuleList()
        current_channels = base_channels
        for rate in upsample_rates:
            out_channels = current_channels // 2
            self.upsample_layers.append(
                nn.ConvTranspose1d(
                    current_channels,
                    out_channels,
                    kernel_size=rate * 2,
                    stride=rate,
                    padding=rate // 2,
                )
            )
            current_channels = out_channels

        self.resblocks = nn.ModuleList(
            [ResidualBlock(current_channels, dilation=3**i) for i in range(3)]
        )

        self.output_conv = nn.Conv1d(
            current_channels, 1, kernel_size=7, padding=3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.norm(self.input_conv(x)), 0.1)

        for upsample in self.upsample_layers:
            x = F.leaky_relu(upsample(x), 0.1)

        for resblock in self.resblocks:
            x = resblock(x)

        return torch.tanh(self.output_conv(x))


class PeriodDiscriminator(nn.Module):
    """Period-based discriminator."""

    def __init__(self, period: int, base_channels: int = 32):
        super().__init__()
        self.period = period
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(1, base_channels, (5, 1), (3, 1), padding=(2, 0)),
                nn.Conv2d(
                    base_channels,
                    base_channels * 2,
                    (5, 1),
                    (3, 1),
                    padding=(2, 0),
                ),
                nn.Conv2d(
                    base_channels * 2,
                    base_channels * 4,
                    (5, 1),
                    (3, 1),
                    padding=(2, 0),
                ),
                nn.Conv2d(
                    base_channels * 4,
                    base_channels * 4,
                    (5, 1),
                    1,
                    padding=(2, 0),
                ),
                nn.Conv2d(base_channels * 4, 1, (3, 1), 1, padding=(1, 0)),
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size, _, time = x.shape
        if time % self.period != 0:
            pad_size = self.period - (time % self.period)
            x = F.pad(x, (0, pad_size))
        x = x.view(batch_size, 1, -1, self.period)

        features = []
        for layer in self.layers[:-1]:
            x = F.leaky_relu(layer(x), 0.1)
            features.append(x)
        x = self.layers[-1](x)
        features.append(x)
        return x.view(batch_size, -1, 1), features


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator."""

    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(period) for period in periods]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        outputs, features = [], []
        for disc in self.discriminators:
            output, feature = disc(x)
            outputs.append(output)
            features.append(feature)
        return outputs, features


class ScaleDiscriminator(nn.Module):
    """Scale discriminator with multiple resolutions."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(1, base_channels, 15, stride=1, padding=7),
                nn.Conv1d(
                    base_channels,
                    base_channels * 2,
                    41,
                    stride=4,
                    padding=20,
                    groups=base_channels,
                ),
                nn.Conv1d(
                    base_channels * 2,
                    base_channels * 4,
                    41,
                    stride=4,
                    padding=20,
                    groups=base_channels * 2,
                ),
                nn.Conv1d(
                    base_channels * 4,
                    base_channels * 8,
                    41,
                    stride=4,
                    padding=20,
                    groups=base_channels * 4,
                ),
                nn.Conv1d(
                    base_channels * 8,
                    base_channels * 16,
                    41,
                    stride=4,
                    padding=20,
                    groups=base_channels * 8,
                ),
                nn.Conv1d(
                    base_channels * 16,
                    base_channels * 32,
                    5,
                    stride=1,
                    padding=2,
                ),
                nn.Conv1d(base_channels * 32, 1, 3, stride=1, padding=1),
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = []
        for layer in self.layers[:-1]:
            x = F.leaky_relu(layer(x), 0.1)
            features.append(x)
        x = self.layers[-1](x)
        features.append(x)
        return x, features


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator."""

    def __init__(self, num_scales: int = 3):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [ScaleDiscriminator() for _ in range(num_scales)]
        )
        self.downsample = nn.AvgPool1d(
            kernel_size=2, stride=2, padding=1, count_include_pad=False
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        outputs, features = [], []
        for disc in self.discriminators:
            output, feature = disc(x)
            outputs.append(output)
            features.append(feature)
            x = self.downsample(x)
        return outputs, features


class SpeechSynthesisGAN(nn.Module):
    """State-of-the-art GAN for speech synthesis."""

    def __init__(self, input_channels: int):
        super().__init__()
        self.generator = Generator(input_channels)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(
        self, mel_spectrogram: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass of the SpeechSynthesisGAN.

        Args:
            mel_spectrogram (torch.Tensor): Input mel spectrogram of shape (batch_size, input_channels, time_steps)

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
                Generated waveform, MPD outputs, and MSD outputs
        """
        waveform = self.generator(mel_spectrogram)
        mpd_outputs, _ = self.mpd(waveform)
        msd_outputs, _ = self.msd(waveform)
        return waveform, mpd_outputs, msd_outputs


def feature_loss(
    real_features: List[List[torch.Tensor]],
    fake_features: List[List[torch.Tensor]],
) -> torch.Tensor:
    """
    Compute the feature matching loss between real and fake features.

    Args:
        real_features (List[List[torch.Tensor]]): Features from the discriminator for real audio
        fake_features (List[List[torch.Tensor]]): Features from the discriminator for generated audio

    Returns:
        torch.Tensor: Feature matching loss
    """
    loss = 0
    for real_feats, fake_feats in zip(real_features, fake_features):
        for real_feat, fake_feat in zip(real_feats, fake_feats):
            loss += F.l1_loss(fake_feat, real_feat)
    return loss


def discriminator_loss(
    real_outputs: List[torch.Tensor], fake_outputs: List[torch.Tensor]
) -> torch.Tensor:
    """
    Compute the discriminator loss.

    Args:
        real_outputs (List[torch.Tensor]): Discriminator outputs for real audio
        fake_outputs (List[torch.Tensor]): Discriminator outputs for generated audio

    Returns:
        torch.Tensor: Discriminator loss
    """
    loss = 0
    for real_output, fake_output in zip(real_outputs, fake_outputs):
        r_loss = torch.mean((1 - real_output) ** 2)
        f_loss = torch.mean(fake_output**2)
        loss += r_loss + f_loss
    return loss


def generator_loss(fake_outputs: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute the generator loss.

    Args:
        fake_outputs (List[torch.Tensor]): Discriminator outputs for generated audio

    Returns:
        torch.Tensor: Generator loss
    """
    loss = 0
    for fake_output in fake_outputs:
        loss += torch.mean((1 - fake_output) ** 2)
    return loss


# Example usage
if __name__ == "__main__":
    input_channels = 80
    batch_size = 4
    time_steps = 128

    model = SpeechSynthesisGAN(input_channels)
    mel_input = torch.randn(batch_size, input_channels, time_steps)

    waveform, mpd_outputs, msd_outputs = model(mel_input)
    print(f"Generated waveform shape: {waveform.shape}")
    print(f"Number of MPD outputs: {len(mpd_outputs)}")
    print(f"Number of MSD outputs: {len(msd_outputs)}")
