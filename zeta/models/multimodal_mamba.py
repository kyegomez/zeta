import torch
from torch import nn, Tensor
from zeta.nn.modules.simple_mamba import Mamba
from zeta import VisualExpert, MLP
from zeta.structs import ViTransformerWrapper, Encoder


class MultiModalMamba(nn.Module):
    """
    MultiModalMamba is a PyTorch module that combines text and image embeddings using a multimodal fusion approach.

    Args:
        dim (int): The dimension of the embeddings.
        depth (int): The depth of the Mamba block.
        dropout (float): The dropout rate.
        heads (int): The number of attention heads.
        d_state (int): The dimension of the state in the Mamba block.
        image_size (int): The size of the input image.
        patch_size (int): The size of the image patches.
        encoder_dim (int): The dimension of the encoder embeddings.
        encoder_depth (int): The depth of the encoder.
        encoder_heads (int): The number of attention heads in the encoder.

    Examples:
    x = torch.randn(1, 16, 64)
    y = torch.randn(1, 3, 64, 64)
    model = MultiModalMamba(
        dim = 64,
        depth = 5,
        dropout = 0.1,
        heads = 4,
        d_state = 16,
        image_size = 64,
        patch_size = 16,
        encoder_dim = 64,
        encoder_depth = 5,
        encoder_heads = 4
    )
    out = model(x, y)
    print(out.shape)

    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        dropout: float,
        heads: int,
        d_state: int,
        image_size: int,
        patch_size: int,
        encoder_dim: int,
        encoder_depth: int,
        encoder_heads: int,
        *args,
        **kwargs,
    ):
        super(MultiModalMamba, self).__init__()
        self.dim = dim
        self.depth = depth
        self.dropout = dropout
        self.heads = heads
        self.d_state = d_state
        self.image_size = image_size
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.encoder_depth = encoder_depth
        self.encoder_heads = encoder_heads

        # Set up the Mamba block
        self.mamba = Mamba(vocab_size, dim, depth)

        # Set up the ViT encoder
        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=encoder_dim, depth=encoder_depth, heads=encoder_heads
            ),
        )

        # Setup the linear layer to project the image embeddings to the same dimension as the text embeddings
        self.linear = nn.Linear(encoder_dim, dim)

        # VisualExpert
        self.fusion_layer = VisualExpert(dim, dim * 2, dropout, heads)

        # MLP
        self.mlp = MLP(dim, dim, expansion_factor=4, depth=1, norm=True)

    def forward(self, text: Tensor, img: Tensor) -> Tensor:
        """
        Forward pass of the MultiModalMamba module.

        Args:
            text (Tensor): The input text embeddings.
            img (Tensor): The input image.

        Returns:
            Tensor: The output embeddings after multimodal fusion.
        """
        encoded_img = self.encoder(img, return_embeddings=True)
        fusion_layer = self.mlp(encoded_img)
        fused = fusion_layer + text
        return self.mamba(fused)


x = torch.randn(1, 16, 64)
y = torch.randn(1, 3, 64, 64)
model = MultiModalMamba(
    vocab_size=16,
    dim=64,
    depth=5,
    dropout=0.1,
    heads=4,
    d_state=16,
    image_size=64,
    patch_size=16,
    encoder_dim=64,
    encoder_depth=5,
    encoder_heads=4,
)
out = model(x, y)
print(out.shape)
