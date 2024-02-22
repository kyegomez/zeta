from torch import Tensor, nn


class VerboseExecution(nn.Module):
    """
    A wrapper class that adds verbosity to the execution of a given model.

    Args:
        model (nn.Module): The model to be executed.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        for name, layer in self.model.named_children():
            for name, layer in self.model.named_children():
                layer.__name__ = name
                layer.register_forward_hook(
                    lambda layer, _, output: print(
                        f"{layer.__name__} output: {output.shape}"
                    )
                )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
