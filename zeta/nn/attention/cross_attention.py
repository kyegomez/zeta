from zeta.nn.architecture.transformer import AttentionLayers

class CrossAttend(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(cross_attend=True,
                         only_cross=True,
                         **kwargs)