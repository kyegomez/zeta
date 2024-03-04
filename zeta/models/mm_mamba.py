import torch
from torch import Tensor, nn

from zeta.nn.modules.mlp import MLP
from zeta.nn.modules.rms_norm import RMSNorm
from zeta.nn.modules.simple_mamba import MambaBlock
from zeta.nn.modules.visual_expert import VisualExpert
from zeta.structs.transformer import Encoder, ViTransformerWrapper


class MultiModalMamba(nn.Module):
    """
    MultiModalMamba model.

    Args:
        vocab_size (int): Size of the vocabulary.
        dim (int): Dimension of the dense vectors.
        depth (int): Number of layers in the model.
        dropout (float): Dropout probability.
        heads (int): Number of attention heads.
        d_state (int): Dimension of the state.
        image_size (int): Size of the input image.
        patch_size (int): Size of the image patch.
        encoder_dim (int): Dimension of the encoder.
        encoder_depth (int): Number of layers in the encoder.
        encoder_heads (int): Number of attention heads in the encoder.
        fusion_method (str): Fusion method to use. Defaults to "mlp", can be one of "mlp", "concat", "add", "visual_expert", "matmul", "mobilevlm", "CrossAttention".
        return_embeddings (bool): Whether to return the embeddings or not. Defaults to False.
        expansion_ratio (int): Expansion ratio for the hidden dimension. Defaults to 4.
        post_fuse_norm (bool): Whether to apply layer normalization after the fusion or not. Defaults to True.

        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Examples::
    import torch
    from mm_mamba.model import MMM

    x = torch.randint(0, 10000, (1, 224))
    img = torch.randn(1, 3, 224, 224)

    model = MMM(
        vocab_size=10000,
        dim=512,
        depth=6,
        dropout=0.1,
        heads=8,
        d_state=512,
        image_size=224,
        patch_size=16,
        encoder_dim=512,
        encoder_depth=6,
        encoder_heads=8,
    )

    out = model(x, img)
    print(out.shape)

    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        dropout: float =0.1,
        heads: int,
        d_state: int,
        image_size: int,
        patch_size: int,
        encoder_dim: int,
        encoder_depth: int,
        encoder_heads: int,
        fusion_method: str = "mlp",
        return_embeddings: bool = False,
        expansion_ratio: int = 4,
        post_fuse_norm: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
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
        self.return_embeddings = return_embeddings
        self.expansion_ratio = expansion_ratio
        self.post_fuse_norm = post_fuse_norm

        # Transforms integer indices to dense vectors of fixed size
        self.embedding = nn.Embedding(vocab_size, dim)

        # MultiModalMambaBlock in a list
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    dim,
                    depth,
                    d_state,
                    expansion_ratio,
                    *args,
                    **kwargs,
                )
            ]
        )

        # Normalization layer
        self.rmsnorm = RMSNorm(dim)
        self.norm = nn.LayerNorm(dim)

        # Linear layer
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

        # Projection for the img
        self.img_proj = nn.Linear(dim, dim)

        # Hidden dim
        self.hidden_dim = dim * expansion_ratio

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
        Forward pass of the MultiModalMamba model.

        Args:
            text (Tensor): Input text tensor.
            img (Tensor): Input image tensor.

        Returns:
            Tensor: Output logits.
        """
        x = self.embedding(text)
        # print(f"Text shape: {x.shape} inside the MMM")

        # Encode the image, Returns the same shape as text
        encoded_img = self.encoder(img, return_embeddings=True)
        # print(f"Image shape: {encoded_img.shape} inside the MMM")
        # Project the image embeddings to the same dimension as the text embeddings
        # We need to project the 2nd dim of the image embeddings to the same dimension as the text embeddings

        # if the fusion method is mlp, use the mlp to fuse the text and image embeddings
        if self.fusion_method == "mlp":
            fusion_layer = self.mlp(encoded_img)
            fused = fusion_layer + x

            if self.post_fuse_norm:
                fused = self.norm(fused)

        # If fusion method is concat, concatenate the text and image embeddings
        if self.fusion_method == "concat":
            fused = torch.concat([x, encoded_img], dim=1)

            if self.post_fuse_norm:
                fused = self.norm(fused)

        if self.fusion_method == "add":
            fused = encoded_img + x

            if self.post_fuse_norm:
                fused = self.norm(fused)

        if self.fusion_method == "visual_expert":
            concat = torch.cat([x, encoded_img], dim=1)
            fused = self.visual_expert(concat)

            if self.post_fuse_norm:
                fused = self.norm(fused)

        if self.fusion_method == "matmul":
            fused = torch.matmul(encoded_img, x)

            if self.post_fuse_norm:
                fused = self.norm(fused)

        # Need to implement this
        if self.fusion_method == "mobilevlm":
            pass

        # Need to implement this
        if self.fusion_method == "CrossAttention":
            pass

        x = fused

        for layer in self.layers:
            x = layer(x) + x

        if self.return_embeddings:
            return x
        else:
            x = self.norm(x)
            logits = self.lm_head(x)

            return logits
