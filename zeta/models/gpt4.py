import torch
from torch import nn, Tensor

from zeta.structs.auto_regressive_wrapper import AutoregressiveWrapper
from zeta.structs.transformer import (
    Decoder,
    Encoder,
    Transformer,
    ViTransformerWrapper,
)


class GPT4(nn.Module):
    """
    GPT4 is a transformer-based model architecture. It initializes with
    a Transformer and AutoregressiveWrapper with default or user-specified parameters.
        Initialize the model with specified or default parameters.
        Args:
        - num_tokens: Number of tokens in the vocabulary
        - max_seq_len: Maximum sequence length
        - dim: Dimension of the model
        - depth: Depth of the model
        - dim_head: Dimension of the model head
        - heads: Number of heads
        - use_abs_pos_emb: Whether to use absolute position embedding
        - alibi_pos_bias: Alibi position bias
        - alibi_num_heads: Number of alibi heads
        - rotary_xpos: Rotary position
        - attn_flash: Attention flash
        - deepnorm: Deep normalization
        - shift_tokens: Number of tokens to shift
        - attn_one_kv_head: Attention one key/value head
        - qk_norm: Query-key normalization
        - attn_qk_norm: Attention query-key normalization
        - attn_qk_norm_dim_scale: Attention query-key normalization dimension scale
        - embedding_provider: Embedding provider module
    """
    def __init__(
        self,
        num_tokens=50432,
        max_seq_len=8192,
        dim=2560,
        depth=32,
        dim_head=128,
        heads=24,
        use_abs_pos_emb=False,
        alibi_pos_bias=True,
        alibi_num_heads=12,
        rotary_xpos=True,
        attn_flash=True,
        attn_one_kv_head=True,  # multiquery attention
        qk_norm=True,
        attn_qk_norm=True,
        attn_qk_norm_dim_scale=True,
        *args,
        **kwargs
    ):
        super().__init__()

        try:
            self.decoder = Transformer(
                num_tokens=num_tokens,
                max_seq_len=max_seq_len,
                use_abs_pos_emb=use_abs_pos_emb,
                attn_layers=Decoder(
                    dim=dim,
                    depth=depth,
                    dim_head=dim_head,
                    heads=heads,
                    alibi_pos_bias=alibi_pos_bias,
                    alibi_num_heads=alibi_num_heads,
                    rotary_xpos=rotary_xpos,
                    attn_flash=attn_flash,
                    attn_one_kv_head=attn_one_kv_head,
                    qk_norm=qk_norm,
                    attn_qk_norm=attn_qk_norm,
                    attn_qk_norm_dim_scale=attn_qk_norm_dim_scale,
                    *args,
                    **kwargs
                ),
            )

            self.decoder = AutoregressiveWrapper(self.decoder)

        except Exception as e:
            print("Failed to initialize Andromeda: ", e)
            raise

    def forward(self, text: Tensor, **kwargs):
        try:
            model_input = self.decoder.forward(text)[0]
            return self.decoder(model_input, padded_x=model_input[0])
        except Exception as e:
            print("Failed in forward method: ", e)
            raise


class GPT4MultiModal(torch.nn.Module):
    """
    GPT4MultiModal is a multi-modal transformer model that combines image and text inputs.

    Args:
        image_size (int): The size of the input image (default: 256).
        patch_size (int): The size of each image patch (default: 32).
        encoder_dim (int): The dimension of the encoder layers (default: 512).
        encoder_depth (int): The number of encoder layers (default: 6).
        encoder_heads (int): The number of attention heads in the encoder (default: 8).
        num_tokens (int): The number of tokens in the vocabulary (default: 20000).
        max_seq_len (int): The maximum sequence length for the decoder (default: 1024).
        decoder_dim (int): The dimension of the decoder layers (default: 512).
        decoder_depth (int): The number of decoder layers (default: 6).
        decoder_heads (int): The number of attention heads in the decoder (default: 8).
        alibi_num_heads (int): The number of attention heads for the alibi mechanism (default: 4).
        use_abs_pos_emb (bool): Whether to use absolute positional embeddings (default: False).
        cross_attend (bool): Whether to enable cross-attention between encoder and decoder (default: True).
        alibi_pos_bias (bool): Whether to use positional bias for the alibi mechanism (default: True).
        rotary_xpos (bool): Whether to use rotary positional embeddings (default: True).
        attn_flash (bool): Whether to use attention flash (default: True).
        qk_norm (bool): Whether to normalize the query-key dot product (default: True).
    """

    def __init__(
        self,
        image_size=256,
        patch_size=32,
        encoder_dim=512,
        encoder_depth=6,
        encoder_heads=8,
        num_tokens=20000,
        max_seq_len=1024,
        decoder_dim=512,
        decoder_depth=6,
        decoder_heads=8,
        alibi_num_heads=4,
        use_abs_pos_emb=False,
        cross_attend=True,
        alibi_pos_bias=True,
        rotary_xpos=True,
        attn_flash=True,
        qk_norm=True,
        *args,
        **kwargs
    ):
        super(GPT4MultiModal, self).__init__()
        
        # Encoder
        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=encoder_dim, depth=encoder_depth, heads=encoder_heads
            ),
        )
        
        # Decoder
        self.decoder = Transformer(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            use_abs_pos_emb=use_abs_pos_emb,
            attn_layers=Decoder(
                dim=decoder_dim,
                depth=decoder_depth,
                heads=decoder_heads,
                cross_attend=cross_attend,
                alibi_pos_bias=alibi_pos_bias,
                alibi_num_heads=alibi_num_heads,
                rotary_xpos=rotary_xpos,
                attn_flash=attn_flash,
                qk_norm=qk_norm,
            ),
        )

    def forward(self, img: Tensor, text: Tensor):
        """
        Performs the forward pass of the GPT4 model.

        Args:
            img (Tensor): The input image tensor.
            text (Tensor): The input text tensor.

        Returns:
            Tensor: The output tensor of the model.
        """
        try:
            encoded = self.encoder(img, return_embeddings=True)
            return self.decoder(text, context=encoded)
        except Exception as error:
            print(f"Failed in forward method: {error}")
            raise
