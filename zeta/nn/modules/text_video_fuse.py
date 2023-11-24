import torch
import torch.nn as nn
from einops import repeat


class TextVideoAttentionFusion(nn.Module):
    """
    Text-Video Attention Fusion

    Args:
        text_features (int): Text features
        video_features (int): Video features

    Shape:
        - text: :math:`(B, S_T, T_F)`
        - video: :math:`(B, S_V, H*W, V_F)`
        - Output: :math:`(B, S_T, S_V, V_F)`

    Examples::
        >>> from zeta.nn.modules import TextVideoAttentionFusion
        >>> import torch
        >>> x = torch.rand(1, 10, 512)
        >>> y = torch.rand(1, 20, 196, 512)
        >>> model = TextVideoAttentionFusion(512, 512)
        >>> out = model(x, y)
        >>> print(out.shape)
        torch.Size([1, 10, 20, 512])

    """

    def __init__(self, text_features, video_features):
        super(TextVideoAttentionFusion, self).__init__()

        # A linear layer for calculating attention scores
        self.linear = nn.Linear(text_features + video_features, 1)

    def forward(self, text, video):
        """forward method"""
        # text: [batch_size, seq_len_text, text_features]
        # video: [batch_size, seq_len_video, h*w, video_features]

        # Get the dimensions
        batch_size, seq_len_video, hw, video_features = video.shape
        _, seq_len_text, text_features = text.shape

        # Using einops to repeat the tensors for matching dimensions
        text_expanded = repeat(
            text, "b st tf -> b st sv hw tf", sv=seq_len_video, hw=hw
        )
        video_expanded = repeat(
            video, "b sv hw vf -> b st sv hw vf", st=seq_len_text
        )

        # Concatenating expanded text tensor and video tensor
        concat_features = torch.cat(
            [text_expanded, video_expanded], dim=-1
        )  # [batch_size, seq_len_text, seq_len_video, h*w, text_features + video_features]

        # Computing attention scores and weights
        attention_scores = self.linear(concat_features).squeeze(
            -1
        )  # [batch_size, seq_len_text, seq_len_video, h*w]
        attention_weights = (
            torch.nn.functional.softmax(attention_scores.flatten(2), dim=-1)
            .view(batch_size, seq_len_text, seq_len_video, hw)
            .unsqueeze(-1)
        )  # [batch_size, seq_len_text, seq_len_video, h*w, 1]

        # Using einsum for the weighted sum across sequence lengths and spatial dimensions
        fused = torch.einsum(
            "btvhj,btvhj->btvj", video_expanded, attention_weights
        )  # [batch_size, seq_len_text, seq_len_video, video_features]

        return fused


# x = torch.rand(1, 10, 512)
# y = torch.rand(1, 20, 196, 512)
# model = TextVideoAttentionFusion(512, 512)
# out = model(x, y)
# print(out)
