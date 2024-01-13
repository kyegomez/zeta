import torch
from torch import nn, Tensor
from zeta.nn.modules.visual_expert import VisualExpert
from zeta.nn.modules.mlp import MLP
from zeta.nn.modules.simple_mamba import MambaBlock
from zeta.structs import ViTransformerWrapper, Encoder


class MultiModalMambaBlock(nn.Module):
    """
    MultiModalMambaBlock is a PyTorch module that combines text and image embeddings using a multimodal fusion approach.

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
        fusion_method (str): The multimodal fusion method to use. Can be one of ["mlp", "concat", "add"].

    Examples:
    x = torch.randn(1, 16, 64)
    y = torch.randn(1, 3, 64, 64)
    model = MultiModalMambaBlock(
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
        fusion_method: str = "mlp",
        expansion_rate: int = 2,
        *args,
        **kwargs,
    ):
        super(MultiModalMambaBlock, self).__init__()
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
        self.fusion_method = fusion_method

        # Hidden dim
        self.hidden_dim = dim * expansion_rate

        # Set up the Mamba block
        self.mamba = MambaBlock(
            dim=dim, depth=depth, d_state=d_state, *args, **kwargs
        )

        # Set up the ViT encoder
        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=encoder_dim,
                depth=encoder_depth,
                heads=encoder_heads,
            ),
        )

        # Setup the linear layer to project the image embeddings to the same dimension as the text embeddings
        self.linear = nn.Linear(encoder_dim, dim)

        # VisualExpert
        self.visual_expert = VisualExpert(dim, self.hidden_dim, dropout, heads)

        # MLP
        self.mlp = MLP(dim, dim, expansion_factor=4, depth=1, norm=True)

    def forward(self, text: Tensor, img: Tensor) -> Tensor:
        """
        Forward pass of the MultiModalMambaBlock module.

        Args:
            text (Tensor): The input text embeddings.
            img (Tensor): The input image.

        Returns:
            Tensor: The output embeddings after multimodal fusion.
        """
        # Encode the image, Returns the same shape as text
        encoded_img = self.encoder(img, return_embeddings=True)
        # print(f"Image shape: {encoded_img.shape} inside the MultiModalMambaBlock")
        # Project the image embeddings to the same dimension as the text embeddings
        # We need to project the 2nd dim of the image embeddings to the same dimension as the text embeddings

        # if the fusion method is mlp, use the mlp to fuse the text and image embeddings
        if self.fusion_method == "mlp":
            fusion_layer = self.mlp(encoded_img)
            fused = fusion_layer + text

        # If fusion method is concat, concatenate the text and image embeddings
        if self.fusion_method == "concat":
            fused = torch.concat([text, encoded_img], dim=1)

        if self.fusion_method == "add":
            fused = encoded_img + text

        if self.fusion_method == "visual_expert":
            concat = torch.cat([text, encoded_img], dim=1)
            fused = self.visual_expert(concat)

        return self.mamba(fused)

    def check_fusion_method(self):
        print("""[mlp] [visualexpert] [projection] [concat] [add] """)
        print(f"""Current fusion method: {self.fusion_method}""")
