from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from zeta.nn.modules.image_to_text import img_to_text


@dataclass
class GILLMapper(nn.Module):
    """
    GILLMapper is a module that maps image and text embeddings using a Transformer model.
    From the paper: "https://arxiv.org/pdf/2305.17216.pdf"

    Args:
        img_emb_size (int): The size of the image embeddings.
        text_emb_size (int): The size of the text embeddings.
        num_encoder_layers (int): The number of layers in the encoder of the Transformer model.
        num_decoder_layers (int): The number of layers in the decoder of the Transformer model.
        heads (int): The number of attention heads in the Transformer model.
        dim_ffn (int): The size of the feed-forward neural network in the Transformer model.
        seq_length (int): The length of the input sequence.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        args (dict, optional): Additional arguments. Defaults to None.

    Example:
        >>> model = GILLMapper(
        ...     img_emb_size=512,
        ...     text_emb_size=512,
        ...     num_encoder_layers=6,
        ...     num_decoder_layers=6,
        ...     heads=8,
        ...     dim_ffn=2048,
        ...     seq_length=100,
        ...     dropout=0.1,
        ...     args=None
        ... )
        >>> img = torch.randn(1, 3, 224, 224)
        >>> text = torch.randn(1, 100, 512)
        >>> out = model(img, text)
        >>> out.shape
    """

    img_emb_size: int
    text_emb_size: int
    num_encoder_layers: int
    num_decoder_layers: int
    heads: int
    dim_ffn: int
    seq_length: int
    dropout: float = 0.1
    args: dict = None

    def __post_init__(self):
        super(GILLMapper, self).__init__()
        self.transformer = nn.Transformer(
            d_model=self.text_emb_size,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_ffn,
        )
        self.img_to_text_proj = nn.Linear(self.img_emb_size, self.text_emb_size)
        self.learned_queries = nn.Parameter(
            torch.randn(self.seq_length, self.text_emb_size)
        )
        self.output_layer = nn.Linear(self.text_emb_size, self.text_emb_size)
        self.text_embedding_layer = nn.Embedding(
            self.seq_length, self.text_emb_size
        )
        self.img_embedding_layer = nn.Linear(
            self.img_emb_size, self.text_emb_size
        )

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.text_emb_size,
                nhead=self.heads,
                dim_feedforward=self.dim_ffn,
            ),
            num_layers=self.num_encoder_layers,
        )

    def forward(self, img: Tensor, text: Tensor) -> Tensor:
        """
        Forward pass of the GILLMapper module.

        Args:
            img (Tensor): The input image tensor. 4D tensor of shape (B, C, H, W).
            text (Tensor): The input text tensor. 3D tensor of shape (batch_size, seq_length).

        Returns:
            Tensor: The output tensor.
        """
        # Embed the image and text
        # img = self.img_embedding_layer(img)
        text = self.text_embedding_layer(text)

        t_b, t_n, t_d = text.shape
        img = img_to_text(img, t_n, t_d)

        # Transforming the img with the encoder
        img = self.transformer_encoder(img)
        print(f"img shape: {img.shape}")

        # Rearrange embeddings for transformer
        img = rearrange(img, "b n d -> n b d ")
        text = rearrange(text, "b n d -> n b d ")

        # Expand learned queries to match the batch
        queries = rearrange(self.learned_queries, "n d -> n 1 d").expand(
            -1, img.shape[1], -1
        )

        # Transformer
        output = self.transformer(src=img, tgt=queries + text)

        # Output layer
        out = self.output_layer(output)
        out = rearrange(out, "n b d -> b n d")

        return out


# Image and text tensors
img = torch.randn(1, 3, 224, 224)
text = torch.randn(1, 100, 512)

# Model Initialization
model = GILLMapper(
    img_emb_size=512,
    text_emb_size=512,
    num_encoder_layers=6,
    num_decoder_layers=6,
    heads=8,
    dim_ffn=2048,
    seq_length=100,
    dropout=0.1,
    args=None,
)

# Forward pass
out = model(img, text)

# Print output shape
print(out.shape)
