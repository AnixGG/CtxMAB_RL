from BaseEnv import BaseEnv
from sklearn.datasets import load_iris
import numpy as np


class IrisEnv(BaseEnv):
    def __init__(self):
        self.data = load_iris()
        self.indexes = np.arange(self.data.data.shape[0])
        self.n_arms = len(np.unique(self.data.target))

    def get_sample(self):
        random_idx = int(np.random.choice(self.indexes))
        current_context = self.data.data[random_idx]
        self.current_target = self.data.target[random_idx]
        return current_context.squeeze()

    def check(self, arm):
        reward = 1 if (arm == self.current_target) else 0
        return reward

    def get_params(self):
        params = {
            "dim": len(self.data.data[0]),
            "n_arms": self.n_arms,
        }
        return params
