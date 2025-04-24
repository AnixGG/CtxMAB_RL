from BaseBandit import BaseBandit
import numpy as np


class UCB2(BaseBandit):
    def __init__(self, n_arms: int, alpha: float = 0.1, *args, **kwargs):
        super().__init__(n_arms)
        self.alpha = alpha

        self.q_values = np.zeros(self.n_arms)
        self.count_q_arms = np.zeros(self.n_arms)
        self.total_steps = 0

    def select_arm(self, *args, **kwargs) -> int:
        best_ucb = -float("inf")
        best_arm = 0
        for arm in range(self.n_arms):
            if self.count_q_arms[arm] == 0:
                return arm
            ucb_arm = self.q_values[arm] + np.sqrt(
                2 * (1 + self.alpha) * np.log(self.total_steps + 1) / self.count_q_arms[arm])
            if ucb_arm > best_ucb:
                best_ucb = ucb_arm
                best_arm = arm
        return best_arm

    def update(self, arm, reward, *args, **kwargs):
        self.count_q_arms[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.count_q_arms[arm]

    def get_name(self):
        return f"UCB2(alpha={self.alpha})"


class UCB(UCB2):
    def __init__(self, n_arms: int, *args, **kwargs):
        super().__init__(n_arms, alpha=0)

    def get_name(self):
        return f"UCB"
