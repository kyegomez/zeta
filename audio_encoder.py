import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class AudioEncoder(nn.Module):
    """
    Audio Encoder class that processes audio input through a Mel Filter Bank, CNN downsampling layers,
    and a Transformer encoder. The output is then passed through a simple two-layer MLP to encode
    each 2 seconds of audio input into 25 tokens.

    Args:
        n_mels (int): Number of mel frequency bins. Default is 128.
        cnn_channels (int): Number of channels in the CNN layers. Default is 64.
        transformer_layers (int): Number of layers in the Transformer. Default is 24.
        nhead (int): Number of heads in the multiheadattention models. Default is 8.
        dim_feedforward (int): The dimension of the feedforward network model in nn.TransformerEncoder. Default is 1024.
        audio_length (int): Length of the input audio in seconds. Default is 2.
        mlp_hidden_dim (int): Dimension of the hidden layer in the MLP. Default is 256.
        output_dim (int): Dimension of the output tokens. Default is 25.
    """

    def __init__(
        self,
        n_mels: int = 128,
        cnn_channels: int = 64,
        transformer_layers: int = 24,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        audio_length: int = 2,
        mlp_hidden_dim: int = 256,
        output_dim: int = 25,
    ):
        super(AudioEncoder, self).__init__()

        self.mel_spectrogram = MelSpectrogram(sample_rate=16000, n_mels=n_mels)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                cnn_channels,
                cnn_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                cnn_channels * 2,
                cnn_channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                cnn_channels * 4,
                cnn_channels * 8,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
        )

        transformer_encoder_layer = TransformerEncoderLayer(
            d_model=cnn_channels * 8,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.transformer_encoder = TransformerEncoder(
            transformer_encoder_layer, num_layers=transformer_layers
        )

        self.mlp = nn.Sequential(
            nn.Linear(cnn_channels * 8, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, output_dim),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AudioEncoder.

        Args:
            audio (torch.Tensor): Input audio tensor of shape (batch_size, num_samples).

        Returns:
            torch.Tensor: Encoded audio tensor of shape (batch_size, num_tokens, output_dim).
        """
        # Convert raw audio to Mel Spectrogram
        mel_spec = self.mel_spectrogram(audio).unsqueeze(
            1
        )  # Add channel dimension

        # Pass through CNN layers
        cnn_out = self.cnn(mel_spec)

        # Flatten CNN output for transformer
        batch_size, channels, height, width = cnn_out.size()
        cnn_out = cnn_out.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)

        # Pass through Transformer
        transformer_out = self.transformer_encoder(cnn_out)

        # Pass through MLP
        output = self.mlp(transformer_out)

        return output


# Example usage:
if __name__ == "__main__":
    # Assume 2 seconds of audio with 16kHz sample rate
    audio_input = torch.randn(8, 32000)  # batch_size = 8, num_samples = 32000

    model = AudioEncoder()
    output = model(audio_input)
    print(output)  # Should output (batch_size, num_tokens, output_dim)
