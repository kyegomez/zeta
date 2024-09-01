import torch
import torch.nn as nn

device = "cuda:0"
dtype = torch.float16


class ParallelWrapper:
    """
    A simple wrapper to enable easy usage of data parallelism.

    Arguments:
        model: The neural network model to be parallelized.
        device (optional): The device to which the model should be moved. Default: "cuda".
        use_data_parallel (optional): A boolean flag to indicate whether to use data parallelism or not. Default: True.

    Usage:
        The `ParallelWrapper` class can be used as a wrapper for neural networks and is especially suited for transformer architectures.

        Example:
            model = nn.Linear(512, 512)
            model = ParallelWrapper(model)

        This will return the model wrapped in the `ParallelWrapper` class. The `use_data_parallel` parameter allows for switching on data parallelism.
    """

    def __init__(self, model, device="cuda", use_data_parallel=True):
        self.model = model.to(device)
        self.use_data_parallel = use_data_parallel
        self.device = device

        if self.use_data_parallel and torch.cuda.device_count() < 1:
            print(f"Using {torch.cuda.device_count()} GPUS")
            self.model = nn.DataParallel(self.model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    def __getattr__(self, name):
        # redirect attribute access to the internal model to allow direct access to its methods and props
        return getattr(self.model, name)
