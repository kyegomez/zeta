from accelerate import Accelerator




def print_num_params(model, accelerator: Accelerator):
    # n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Number of parameters in model: {n_params}")

