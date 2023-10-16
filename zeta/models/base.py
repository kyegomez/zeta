from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def forward(self):
        pass
