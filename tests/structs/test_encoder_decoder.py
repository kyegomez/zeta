from argparse import Namespace

import torch

from zeta.structs.encoder_decoder import EncoderDecoder


def test_encoder_decoder_initialization():
    args = Namespace(share_all_embeddings=True)
    encoder_decoder = EncoderDecoder(args)

    assert isinstance(encoder_decoder, EncoderDecoder)
    assert encoder_decoder.args == args
    assert encoder_decoder.args.share_all_embeddings is True
    assert encoder_decoder.args.share_decoder_input_output_embed is True


def test_encoder_decoder_forward():
    args = Namespace(share_all_embeddings=True)
    encoder_decoder = EncoderDecoder(args)

    src_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    prev_output_tokens = torch.tensor([[7, 8, 9], [10, 11, 12]])

    output = encoder_decoder(src_tokens, prev_output_tokens)

    assert isinstance(output, torch.Tensor)
    assert output.shape == prev_output_tokens.shape


def test_encoder_decoder_forward_features_only():
    args = Namespace(share_all_embeddings=True)
    encoder_decoder = EncoderDecoder(args)

    src_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    prev_output_tokens = torch.tensor([[7, 8, 9], [10, 11, 12]])

    output = encoder_decoder(src_tokens, prev_output_tokens, features_only=True)

    assert isinstance(output, torch.Tensor)
    assert output.shape == prev_output_tokens.shape
