# Copyright (c) 2022 Agora
# Licensed under The MIT License [see LICENSE for details]

import torch.nn as nn

from zeta.structs.transformer import Decoder, Encoder


class EncoderDecoder(nn.Module):
    """
    A module that combines an encoder and a decoder for sequence-to-sequence tasks.

    Args:
        args (argparse.Namespace): The arguments passed to the module.
        encoder_embed_tokens (torch.Tensor, optional): The input embeddings for the encoder. Defaults to None.
        encoder_embed_positions (torch.Tensor, optional): The positions of the encoder input embeddings. Defaults to None.
        decoder_embed_tokens (torch.Tensor, optional): The input embeddings for the decoder. Defaults to None.
        decoder_embed_positions (torch.Tensor, optional): The positions of the decoder input embeddings. Defaults to None.
        output_projection (torch.Tensor, optional): The projection layer for the decoder output. Defaults to None.
        **kwargs: Additional keyword arguments.

    Attributes:
        args (argparse.Namespace): The arguments passed to the module.
        encoder (Encoder): The encoder module.
        decoder (Decoder): The decoder module.
    """

    def __init__(
        self,
        args,
        encoder_embed_tokens=None,
        encoder_embed_positions=None,
        decoder_embed_tokens=None,
        decoder_embed_positions=None,
        output_projection=None,
        **kwargs,
    ):
        super().__init__()
        self.args = args
        if args.share_all_embeddings:
            args.share_decoder_input_output_embed = True

        self.encoder = Encoder(
            args,
            encoder_embed_tokens,
            encoder_embed_positions,
            is_encoder_decoder=True,
            **kwargs,
        )

        if args.share_all_embeddings and decoder_embed_tokens is None:
            decoder_embed_tokens = self.encoder.embed_tokens

        self.decoder = Decoder(
            args,
            decoder_embed_tokens,
            decoder_embed_positions,
            output_projection,
            is_encoder_decoder=True,
            **kwargs,
        )

    def forward(
        self,
        src_tokens,
        prev_output_tokens,
        return_all_hiddens=False,
        features_only=False,
        **kwargs,
    ):
        """
        Forward pass of the EncoderDecoder module.

        Args:
            src_tokens (torch.Tensor): The source tokens.
            prev_output_tokens (torch.Tensor): The previous output tokens.
            return_all_hiddens (bool, optional): Whether to return all hidden states. Defaults to False.
            features_only (bool, optional): Whether to return only the features. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            decoder_out (torch.Tensor): The output of the decoder module.
        """
        encoder_out = self.encoder(
            src_tokens, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out
