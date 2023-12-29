from abc import ABC


class BaseModel(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def forward(self):
        raise NotImplementedError
