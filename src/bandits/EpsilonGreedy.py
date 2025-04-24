from BaseBandit import BaseBandit
import numpy as np


class EpsilonGreedy(BaseBandit):
    def __init__(self, n_arms: int, epsilon: float = 0.1, *args, **kwargs):
        super().__init__(n_arms)
        self.epsilon = epsilon

        self.q_values = np.zeros(self.n_arms)
        self.count_q_arms = np.zeros(self.n_arms)

    def select_arm(self, *args, **kwargs):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.q_values)

    def update(self, arm, reward, *args, **kwargs):
        self.count_q_arms[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.count_q_arms[arm]

    def get_name(self):
        return f"EpsilonGreedy(epsilon={self.epsilon})"
