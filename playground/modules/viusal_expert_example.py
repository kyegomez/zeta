import torch
from zeta.nn.modules.visual_expert import VisualExpert

visual_expert = VisualExpert(1024, 2048, 0.1, 16)
x = torch.randn(1, 10, 1024)  # B, SEQ_LEN, DIM

out = visual_expert(x)
print(f"out: {out} out.dtype {out.dtype} out.device {out.device} out.shape{out.shape} ")
