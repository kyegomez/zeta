from zeta.structs.transformer import Transformer, Decoder
from zeta.structs.auto_regressive_wrapper import AutoregressiveWrapper


class LLama2:
    def __init__(
        self,
        num_tokens=50432,
        max_seq_len=8192,
        dim=2560,
        depth=32,
        dim_head=128,
        heads=24,
        rotary_xpos=True,
        attn_flash=True,
    ):
        super().__init__()

        self.llama2 = Transformer(
            num_tokens=50000,
            max_seq_len=4096,
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                dim_head=dim_head,
                heads=heads,
                attn_flash=attn_flash,
                rotary_xpos=rotary_xpos,
            ),
        )
        self.decoder = AutoregressiveWrapper(self.decoder)

    def forward(self, text):
        model_input = self.decoder.forward(text)[0]
        return self.decoder(model_input, padded_x=model_input[0])
