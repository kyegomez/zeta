import torch
import argparse
import pytest

from zeta.nn import EncoderDecoder, Encoder, Decoder


@pytest.fixture
def encoder_decoder():
    args = argparse.Namespace(share_all_embeddings=True)
    encoder_embed_tokens = torch.Tensor(2, 3)
    encoder_embed_positions = torch.Tensor(2, 3)
    decoder_embed_tokens = torch.Tensor(2, 3)
    decoder_embed_positions = torch.Tensor(2, 3)
    output_projection = torch.Tensor(2, 3)

    return EncoderDecoder(
        args,
        encoder_embed_tokens,
        encoder_embed_positions,
        decoder_embed_tokens,
        decoder_embed_positions,
        output_projection,
    )


def test_initialization(encoder_decoder):
    assert isinstance(encoder_decoder, EncoderDecoder)
    assert isinstance(encoder_decoder.encoder, Encoder)
    assert isinstance(encoder_decoder.decoder, Decoder)


def test_args_share_all_embeddings_propagation(encoder_decoder):
    assert encoder_decoder.args.share_decoder_input_output_embed is True


def test_forward_pass(encoder_decoder):
    src_tokens = torch.Tensor(2, 3)
    prev_output_tokens = torch.Tensor(2, 3)

    output = encoder_decoder.forward(src_tokens, prev_output_tokens)

    assert isinstance(output, torch.Tensor)
