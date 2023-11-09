import torch 
from torch import nn
import torch.nn.functional as F


class ModalityAdaptiveModule(nn.Module):
    """
    Modality Adaptive Module

    Args:
        dim: int
            The dimension of the input features
        heads: int
            The number of heads to use for the attention mechanism

    Returns:
        x: torch.Tensor


    Examples:
        >>> x = torch.randn(1, 3, 512)
        >>> y = torch.randn(1, 3, 512)
        >>> model = ModalityAdaptiveModule(512, 8)
        >>> out = model(x, y)
        >>> print(out.shape)
        torch.Size([1, 3, 512])

    
    """
    def __init__(
        self,
        dim: int,
        heads: int
    ):
        super(ModalityAdaptiveModule, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5
        assert dim % heads == 0, f"dim must alwasy be divisible by heads"

        # Initialize the normalization layers for each modality
        self.norm_text = nn.LayerNorm(dim)
        self.norm_img = nn.LayerNorm(dim)

        # Initialize the img linear layers
        self.img_v_proj = nn.Linear(dim, dim)
        self.img_k_proj = nn.Linear(dim, dim)

        # Initialize the linear layers for the text
        self.text_v_proj = nn.Linear(dim, dim)
        self.text_k_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)

        # Initialize the linear layer
        self.proj = nn.Linear(dim, dim)

    def modality_indicator(self, x):
        """Function that returns the modality indicator"""
        if x.dim() == 4:
            return 0
        elif x.dim() == 3:
            return 1
        else:
            raise ValueError("The tensor must be 3 or 4 dimensions")

        # indicator = nn.Linear(self.dim, self.heads)
        # modality_weights = torch.sigmoid(indicator(x))
        # return modality_weights
    
    def forward(self, text, img):
        """Forward pass of the modality adaptive module"""

        # Normalize the text and image features
        text_normalized = self.norm_text(text)
        img_normalized = self.norm_img(img)

        # Concatenate the normalized text and image features
        norms_concat = torch.concat((text_normalized, img_normalized))

        # Project the text and image features to the same dimension
        vision_v = self.img_v_proj(img_normalized)
        vision_k = self.img_k_proj(img_normalized)
        # Text features are projected to the same dimension as the image features
        text_v = self.text_v_proj(text_normalized)
        text_k = self.text_k_proj(text_normalized)

        # Combine keys from both modalities
        keys_combined = torch.cat((text_k, vision_k))
        values_combined = torch.cat((text_v, vision_v))

        # Project the query to the same dimension as the image and text features
        q = self.q_proj(norms_concat)

        # Matmul between the query and the keys
        matmuled = torch.matmul(q, keys_combined)

        # add scale
        matmul_scale = matmuled * self.scale

        # Attention mechanism: dot product of queries and keys, scaled and normalized
        attn = torch.softmax(matmul_scale)

        # Matmul between the softmaxed matmuled and the values
        x = torch.matmul(attn, values_combined)

        # Projected matmul
        x = self.proj(x)

        # Normalize the outputs
        normed_text = self.norm_text(x)
        normed_img = self.norm_img(x)
        x = torch.concat((normed_text, normed_img))

        return x


x = torch.randn(1, 3, 512)
y = torch.randn(1, 3, 512)

model = ModalityAdaptiveModule(512, 8)

out = model(x, y)

print(out.shape)