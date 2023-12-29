from abc import ABC


class BaseModel(ABC):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError
