import pytest
from zeta.models import Andromeda


@pytest.fixture
def init_andromeda():
    return Andromeda(
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
        attn_kv_heads=2,
        qk_norm=True,
        attn_qk_norm=True,
        attn_qk_norm_dim_scale=True,
    )


def test_initial_parameters(init_andromeda):
    assert init_andromeda.num_tokens == 50432
    assert init_andromeda.max_seq_len == 8192
    assert init_andromeda.dim == 2560
    assert init_andromeda.depth == 32
    assert init_andromeda.dim_head == 128
    assert init_andromeda.heads == 24
    assert init_andromeda.use_abs_pos_emb is False
    assert init_andromeda.alibi_pos_bias is True
    assert init_andromeda.alibi_num_heads == 12
    assert init_andromeda.rotary_xpos is True
    assert init_andromeda.attn_flash is True
    assert init_andromeda.attn_kv_heads == 2
    assert init_andromeda.qk_norm is True
    assert init_andromeda.attn_qk_norm is True
    assert init_andromeda.attn_qk_norm_dim_scale is True


def test_initialization_exception():
    with pytest.raises(Exception):
        Andromeda(num_tokens="wrong_type")


def test_forward_successful(init_andromeda, monkeypatch):

    def mock_forward(self, text_tokens):
        return [text_tokens]

    monkeypatch.setattr("zeta.models.AutoregressiveWrapper.forward",
                        mock_forward)

    result = init_andromeda.forward([1, 2, 3, 4])
    assert result == [1, 2, 3, 4]


def test_forward_exception(init_andromeda, monkeypatch):

    def mock_forward(self, text_tokens):
        raise Exception("Test Forward Error")

    monkeypatch.setattr("zeta.models.AutoregressiveWrapper.forward",
                        mock_forward)

    with pytest.raises(Exception, match="Test Forward Error"):
        init_andromeda.forward([1, 2, 3, 4])
