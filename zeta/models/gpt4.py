from torch.nn import nn

from zeta.nn.architecture.transformer import (
    AndromedaEmbedding,
    AutoregressiveWrapper,
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
    def __init__(self, 
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
                #  shift_tokens=1, 
                 attn_one_kv_head=True,  # multiquery attention
                 qk_norm=True, 
                 attn_qk_norm=True, 
                 attn_qk_norm_dim_scale=True, 
                 embedding_provider=AndromedaEmbedding()):
        super().__init__()

        try:
            self.decoder = Transformer(
                num_tokens=num_tokens,
                max_seq_len=max_seq_len,
                use_abs_pos_emb=use_abs_pos_emb,
                embedding_provider=embedding_provider,
                attn_layers=Decoder(
                    dim=dim,
                    depth=depth,
                    dim_head=dim_head,
                    heads=heads,
                    alibi_pos_bias=alibi_pos_bias,
                    alibi_num_heads=alibi_num_heads,
                    rotary_xpos=rotary_xpos,
                    attn_flash=attn_flash,
                    # deepnorm=deepnorm,
                    # shift_tokens=shift_tokens,
                    attn_one_kv_head=attn_one_kv_head,
                    qk_norm=qk_norm,
                    attn_qk_norm=attn_qk_norm,
                    attn_qk_norm_dim_scale=attn_qk_norm_dim_scale
                )
            )

            self.decoder = AutoregressiveWrapper(self.decoder)

        except Exception as e:
            print("Failed to initialize Andromeda: ", e)
            raise

    def forward(self, text_tokens, **kwargs):
        try:
            model_input = self.decoder.forward(text_tokens)[0]
            return self.decoder(model_input, padded_x=model_input[0])
        except Exception as e:
            print("Failed in forward method: ", e)
            raise



class GPT4MultiModal(nn.Module):
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
    def __init__(self, 
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
                 image_size=256,
                 patch_size=32,
                 attn_one_kv_head=True, 
                 qk_norm=True, 
                 attn_qk_norm=True, 
                 attn_qk_norm_dim_scale=True, 
                 embedding_provider=AndromedaEmbedding()):
        super(GPT4MultiModal).__init__()

        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=dim,
                depth=depth,
                dim_head=dim_head,
                heads=heads
            )
        )

        self.transformer = Transformer(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            use_abs_pos_emb=use_abs_pos_emb,
            embedding_provider=embedding_provider,
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
                cross_attend=True
            )
        )

        self.decoder = AutoregressiveWrapper(self.transformer)

    def forward(self, img, text_tokens, **kwargs):
        """
        Forward pass through the model. It expects the input text_tokens.
        Args:
        - text_tokens: Input tokens
        - kwargs: Other arguments
        Returns:
        - output from the decoder
        """
        try:
            encoded_img = self.encoder(img, return_embeddings=True)
            return self.decoder(text_tokens, context=encoded_img)
        except Exception as error:
            print(f"Failed in forward method: {error}")
            raise