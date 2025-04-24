from abc import ABC, abstractmethod
import numpy as np


class BaseEnv(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_sample(self, *args, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def check(self, *args, **kwargs) -> int:
        pass

    @abstractmethod
    def get_params(self, *args, **kwargs) -> dict:
        pass
