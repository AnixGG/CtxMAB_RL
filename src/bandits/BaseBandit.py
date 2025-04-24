from abc import ABC, abstractmethod
import numpy as np


class BaseBandit(ABC):
    @abstractmethod
    def __init__(self, n_arms: int, *args, **kwargs):
        self.n_arms = n_arms

    @abstractmethod
    def select_arm(self, *args, **kwargs) -> int:
        pass

    @abstractmethod
    def update(self, context: np.ndarray, arm: int, reward: float, *args, **kwargs):
        pass

    @abstractmethod
    def get_name(self):
        pass
