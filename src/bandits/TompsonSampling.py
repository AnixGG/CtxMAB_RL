from BaseBandit import BaseBandit
import numpy as np
from scipy.stats import beta


class TompsonSampling(BaseBandit):
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.beta = np.ones(n_arms)
        self.alpha = np.ones(n_arms)

    def select_arm(self, *args, **kwargs) -> np.ndarray[int]:
        return np.argmax([beta.rvs(a, b) for a, b in zip(self.alpha, self.beta)])

    def update(self, arm: int, reward: float, *args, **kwargs):
        if reward:
            self.alpha[arm] += 1
            return
        self.beta[arm] += 1

    def get_name(self):
        return "ThompsonSampling"
