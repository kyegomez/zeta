from zeta.nn.architecture.attn_layers import AttentionLayers


class Encoder(AttentionLayers):
    def __init__(self,
                 **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal=False, **kwargs)

        