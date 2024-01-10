import torch.nn as nn
from einops import rearrange

class AudioToTextEmbeddings(nn.Module):
    def __init__(self, input_channels, output_dim, seq_len: int, kernel_size=3, stride=1):
        """
        Initializes the module to transform audio tensor to a format similar to text tensor.

        Parameters:
        input_channels (int): Number of input channels in the audio tensor.
        output_dim (int): Desired dimension size for the output tensor.
        kernel_size (int): Kernel size for the convolution layer.
        stride (int): Stride for the convolution layer.
        """
        super(AudioToTextEmbeddings, self).__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.conv1d = nn.Conv1d(input_channels, output_dim, kernel_size, stride=stride)
        self.flatten = nn.Flatten(start_dim=1)  # Flatten all dimensions except batch

    def forward(self, x):
        """
        Forward pass for transforming audio tensor to text-like tensor.

        Parameters:
        x (torch.Tensor): Input 3D audio tensor of shape [B, C, T], where
                          B = Batch size,
                          C = Channels,
                          T = Time frames.

        Returns:
        torch.Tensor: Output 3D tensor of shape [B, T', output_dim], where T' is the 
                      transformed time dimension.
        """
        b, c, t = x.shape
        x = self.conv1d(x)
        # Optionally, additional processing can be done here
        x = self.flatten(x)
        # Reshape to have sequence length as the second dimension
        b, c_t = x.shape
        x = x.view(b, -1, self.conv1d.out_channels)
        
        b, t, c = x.shape
        x = rearrange(x, "b t c -> b c t")
        proj = nn.Linear(t, self.seq_len)
        x = proj(x)
        x = rearrange(x, "b c t -> b t c")
        
        
        return x

# # Example usage:
# # Define the transformer with appropriate input channels and desired output dimension
# audio_transformer = AudioToTextEmbeddings(input_channels=1, output_dim=512, seq_len=1000)
# audio_tensor = torch.randn(1, 1, 16000)  # Example audio tensor (2 samples, 1 channel, 16000 time frames)
# text_like_tensor = audio_transformer(audio_tensor)
# print(text_like_tensor.shape)  # Expected shape: [Batch size, Time frames, 512]
