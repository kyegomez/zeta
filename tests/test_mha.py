from zeta.utils.attention.multihead_attention import MultiheadAttention
import torch
import unittest
from zeta import MultiheadAttention


class TestMultiheadAttention(unittest.TestCase):
    def setUp(self):
        self.args = {"xpos_rel_pos": True, "xpos_scale_base": 2, "layernorm_eps": 1e-5}
        self.embed_dim = 64
        self.num_heads = 4
        self.multihead_attn = MultiheadAttention(
            self.args, self.embed_dim, self.num_heads
        )

    def test_forward_shape(self):
        query = torch.rand(16, 20, self.embed_dim)
        key = torch.rand(16, 20, self.embed_dim)
        value = torch.rand(16, 20, self.embed_dim)
        attn, attn_weights = self.multihead_attn(query, key, value)
        self.assertEqual(attn.shape, (16, 20, self.embed_dim))
        self.assertEqual(attn_weights.shape, (self.num_heads, 16, 20, 20))

    def test_forward_incremental_state(self):
        query = torch.rand(16, 20, self.embed_dim)
        key = torch.rand(16, 20, self.embed_dim)
        value = torch.rand(16, 20, self.embed_dim)
        incremental_state = {
            "prev_key": torch.rand(
                16, self.num_heads, 10, self.embed_dim // self.num_heads
            ),
            "prev_value": torch.rand(
                16, self.num_heads, 10, self.embed_dim // self.num_heads
            ),
        }
        attn, attn_weights = self.multihead_attn(
            query, key, value, incremental_state=incremental_state
        )
        self.assertEqual(attn.shape, (16, 20, self.embed_dim))
        self.assertEqual(attn_weights.shape, (self.num_heads, 16, 20, 30))

    def test_forward_attn_mask(self):
        query = torch.rand(16, 20, self.embed_dim)
        key = torch.rand(16, 20, self.embed_dim)
        value = torch.rand(16, 20, self.embed_dim)
        attn_mask = torch.ones(20, 20)
        attn, attn_weights = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        self.assertEqual(attn.shape, (16, 20, self.embed_dim))
        self.assertEqual(attn_weights.shape, (self.num_heads, 16, 20, 20))

    def test_forward_key_padding_mask(self):
        query = torch.rand(16, 20, self.embed_dim)
        key = torch.rand(16, 20, self.embed_dim)
        value = torch.rand(16, 20, self.embed_dim)
        key_padding_mask = torch.ones(16, 20)
        attn, attn_weights = self.multihead_attn(
            query, key, value, key_padding_mask=key_padding_mask
        )
        self.assertEqual(attn.shape, (16, 20, self.embed_dim))
        self.assertEqual(attn_weights.shape, (self.num_heads, 16, 20, 20))

    def test_forward_rel_pos(self):
        query = torch.rand(16, 20, self.embed_dim)
        key = torch.rand(16, 20, self.embed_dim)
        value = torch.rand(16, 20, self.embed_dim)
        rel_pos = torch.rand(16, self.num_heads, 20, 20)
        attn, attn_weights = self.multihead_attn(query, key, value, rel_pos=rel_pos)
        self.assertEqual(attn.shape, (16, 20, self.embed_dim))
        self.assertEqual(attn_weights.shape, (self.num_heads, 16, 20, 20))

    def test_forward_is_first_step(self):
        query = torch.rand(16, 20, self.embed_dim)
        key = torch.rand(16, 20, self.embed_dim)
        value = torch.rand(16, 20, self.embed_dim)
        attn, attn_weights = self.multihead_attn(query, key, value, is_first_step=True)
        self.assertEqual(attn.shape, (16, 20, self.embed_dim))
        self.assertEqual(attn_weights.shape, (self.num_heads, 16, 20, 20))

    def test_forward_is_not_first_step(self):
        query = torch.rand(16, 20, self.embed_dim)
        key = torch.rand(16, 20, self.embed_dim)
        value = torch.rand(16, 20, self.embed_dim)
        attn, attn_weights = self.multihead_attn(query, key, value, is_first_step=False)
        self.assertEqual(attn.shape, (16, 20, self.embed_dim))
        self.assertEqual(attn_weights.shape, (self.num_heads, 16, 20, 20))

    def test_forward_different_query_key_value_size(self):
        query = torch.rand(16, 20, self.embed_dim)
        key = torch.rand(16, 30, self.embed_dim)
        value = torch.rand(16, 30, self.embed_dim)
        with self.assertRaises(AssertionError):
            self.multihead_attn(query, key, value)

    def test_forward_different_batch_size(self):
        query = torch.rand(16, 20, self.embed_dim)
        key = torch.rand(32, 20, self.embed_dim)
        value = torch.rand(32, 20, self.embed_dim)
        with self.assertRaises(AssertionError):
            self.multihead_attn(query, key, value)

    def test_forward_different_embed_dim(self):
        query = torch.rand(16, 20, 128)
        key = torch.rand(16, 20, 128)
        value = torch.rand(16, 20, 128)
        with self.assertRaises(AssertionError):
            self.multihead_attn(query, key, value)

    def test_forward_no_value(self):
        query = torch.rand(16, 20, self.embed_dim)
        key = torch.rand(16, 20, self.embed_dim)
        with self.assertRaises(AssertionError):
            self.multihead_attn(query, key, None)

    def test_forward_no_key(self):
        query = torch.rand(16, 20, self.embed_dim)
        value = torch.rand(16, 20, self.embed_dim)
        with self.assertRaises(AssertionError):
            self.multihead_attn(query, None, value)

    def test_forward_no_query(self):
        key = torch.rand(16, 20, self.embed_dim)
        value = torch.rand(16, 20, self.embed_dim)
        with self.assertRaises(AssertionError):
            self.multihead_attn(None, key, value)

    def test_forward_no_input(self):
        with self.assertRaises(AssertionError):
            self.multihead_attn(None, None, None)

    def test_forward_zero_length_input(self):
        query = torch.rand(16, 0, self.embed_dim)
        key = torch.rand(16, 0, self.embed_dim)
        value = torch.rand(16, 0, self.embed_dim)
        attn, attn_weights = self.multihead_attn(query, key, value)
        self.assertEqual(attn.shape, (16, 0, self.embed_dim))
        self.assertEqual(attn_weights.shape, (self.num_heads, 16, 0, 0))

    def test_forward_one_length_input(self):
        query = torch.rand(16, 1, self.embed_dim)
        key = torch.rand(16, 1, self.embed_dim)
        value = torch.rand(16, 1, self.embed_dim)
        attn, attn_weights = self.multihead_attn(query, key, value)
        self.assertEqual(attn.shape, (16, 1, self.embed_dim))
        self.assertEqual(attn_weights.shape, (self.num_heads, 16, 1, 1))

    def test_forward_large_input(self):
        query = torch.rand(16, 1000, self.embed_dim)
        key = torch.rand(16, 1000, self.embed_dim)
        value = torch.rand(16, 1000, self.embed_dim)
        attn, attn_weights = self.multihead_attn(query, key, value)
        self.assertEqual(attn.shape, (16, 1000, self.embed_dim))
        self.assertEqual(attn_weights.shape, (self.num_heads, 16, 1000, 1000))


if __name__ == "__main__":
    unittest.main()
