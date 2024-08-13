import torch
import torch.nn as nn

from zeta.nn.quant.lfq import LFQ


def test_lfg_init():
    lfg = LFQ(dim=64, codebook_size=16)
    assert isinstance(lfg, LFQ)
    assert lfg.dim == 64
    assert lfg.codebook_dim == 4
    assert lfg.num_codebooks == 1
    assert lfg.keep_num_codebooks_dim is False
    assert isinstance(lfg.project_in, nn.Linear)
    assert isinstance(lfg.project_out, nn.Linear)
    assert lfg.has_projections is False
    assert isinstance(lfg.activation, nn.Identity)
    assert lfg.diversity_gamma == 1.0
    assert lfg.entropy_loss_weight == 0.1
    assert lfg.codebook_scale == 1.0
    assert lfg.commitment_loss_weight == 0.25
    assert torch.all(lfg.mask == 2 ** torch.arange(3, -1, -1))
    assert lfg.zero == 0.0
    assert torch.all(
        lfg.codebook
        == lfg.bits_to_codes(
            ((torch.arange(16)[..., None].int() & lfg.mask) != 0).float()
        )
    )


def test_lfg_init_custom_params():
    lfg = LFQ(
        dim=128,
        codebook_size=32,
        entropy_loss_weight=0.2,
        commitment_loss_weight=0.3,
        diversity_gamma=2.0,
        straight_through_activation=nn.ReLU(),
        num_codebooks=2,
        keep_num_codebooks_dim=True,
        codebook_scale=2.0,
    )
    assert lfg.dim == 128
    assert lfg.codebook_dim == 5
    assert lfg.num_codebooks == 2
    assert lfg.keep_num_codebooks_dim is True
    assert isinstance(lfg.activation, nn.ReLU)
    assert lfg.diversity_gamma == 2.0
    assert lfg.entropy_loss_weight == 0.2
    assert lfg.codebook_scale == 2.0
    assert lfg.commitment_loss_weight == 0.3
    assert torch.all(lfg.mask == 2 ** torch.arange(4, -1, -1))
    assert torch.all(
        lfg.codebook
        == lfg.bits_to_codes(
            ((torch.arange(32)[..., None].int() & lfg.mask) != 0).float()
        )
    )


def test_lfq_forward():
    lfq = LFQ(dim=64, codebook_size=16)
    x = torch.randn(2, 64)
    output, loss, _, _ = lfq(x)
    assert output.shape == x.shape
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
