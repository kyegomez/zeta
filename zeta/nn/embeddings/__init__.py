# embeddings
from zeta.nn.embeddings.rope import RotaryEmbedding
from zeta.nn.embeddings.xpos_relative_position import XPOS, rotate_every_two, apply_rotary_pos_emb
from zeta.nn.embeddings.multiway_network import MultiwayEmbedding, MultiwayNetwork, MultiwayWrapper
