from BaseBandit import BaseBandit
import numpy as np


# --- Classic LinUCB implementation ---

# class LinUCB(BaseBandit):
#   def __init__(self, n_arms: int, dim: int, alpha: int = 1, *args, **kwargs):
#     super().__init__(n_arms)
#     self.dim = dim
#     self.alpha = alpha
#
#     self.A = [np.eye(self.dim) for _ in range(self.n_arms)]
#     self.b = [np.zeros(self.dim) for _ in range(self.n_arms)]

#   def select_arm(self, context) -> int:
#     best_ucb = -float("inf")
#     for arm in range(self.n_arms):
#       inverse_A = np.linalg.inv(self.A[arm])
#       theta_a = inverse_A @ self.b[arm]

#       ucb_arm = theta_a @ context + self.alpha * np.sqrt(context @ inverse_A @ context)
#       if ucb_arm > best_ucb:
#         best_ucb = ucb_arm
#         best_arm = arm
#     return best_arm

#   def update(self, context, arm, reward):
#     context = np.expand_dims(context, axis=1)
#     self.A[arm] += context @ context.T
#     self.b[arm] += reward * context.squeeze()

#   def get_name(self):
#     return f"LinUCB(alpha={self.alpha})"

# --- LinUCB Sherman-Morrison Optimized ---

class LinUCB(BaseBandit):
    def __init__(self, n_arms: int, dim: int, alpha: float = 1, *args, **kwargs):
        super().__init__(n_arms)
        self.dim = dim
        self.alpha = alpha

        self.A = [np.eye(self.dim) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.dim) for _ in range(self.n_arms)]
        self.A_inv = [np.eye(self.dim) for _ in range(self.n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        best_ucb = -float("inf")
        best_arm = 0
        for arm in range(self.n_arms):
            theta_a = self.A_inv[arm] @ self.b[arm]
            ucb_arm = theta_a @ context + self.alpha * np.sqrt(context @ self.A_inv[arm] @ context)
            if ucb_arm > best_ucb:
                best_ucb = ucb_arm
                best_arm = arm
        return best_arm

    def update(self, context, arm, reward, *args, **kwargs):
        context = np.expand_dims(context, axis=1)
        self.A[arm] += context @ context.T
        self.b[arm] += reward * context.squeeze()
        self.A_inv[arm] -= (self.A_inv[arm] @ (context @ context.T) @ self.A_inv[arm]) / (
                1 + context.T @ self.A_inv[arm] @ context)

    def get_name(self):
        return f"LinUCB(alpha={self.alpha})"


# --- LinUCB with added features - OHE arm ---
class LinUCBoheARMS(BaseBandit):
    def __init__(self, n_arms: int, dim: int, alpha: int = 1, *args, **kwargs):
        super().__init__(n_arms)
        self.dim = dim + self.n_arms
        self.alpha = alpha

        self.A = [np.eye(self.dim) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.dim) for _ in range(self.n_arms)]
        self.A_inv = [np.eye(self.dim) for _ in range(self.n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        best_ucb = -float("inf")
        best_arm = 0
        for arm in range(self.n_arms):
            theta_a = self.A_inv[arm] @ self.b[arm]

            ohe_arm = np.zeros(self.n_arms)
            ohe_arm[arm] = 1
            expanded_context = np.concatenate([context, ohe_arm])

            ucb_arm = theta_a @ expanded_context + self.alpha * np.sqrt(
                expanded_context @ self.A_inv[arm] @ expanded_context)
            if ucb_arm > best_ucb:
                best_ucb = ucb_arm
                best_arm = arm
        return best_arm

    def update(self, context, arm, reward, *args, **kwargs):
        ohe_arm = np.zeros(self.n_arms)
        ohe_arm[arm] = 1
        expanded_context = np.concatenate([context, ohe_arm])

        context = np.expand_dims(expanded_context, axis=1)
        self.A[arm] += context @ context.T
        self.A_inv[arm] -= (self.A_inv[arm] @ (context @ context.T) @ self.A_inv[arm]) / (
                1 + context.T @ self.A_inv[arm] @ context)
        self.b[arm] += reward * context.squeeze()

    def get_name(self):
        return f"LinUCBoheARMS(alpha={self.alpha})"


# --- LinUCB with adaptive alpha ---
class AdaptiveLinUCB(LinUCB):
    def __init__(self, n_arms: int, dim: int, alpha: float = 1, *args, **kwargs):
        super().__init__(n_arms, dim, alpha)
        self.first_alpha = self.alpha
        self.total_steps = 0

    def update(self, context, arm, reward, *args, **kwargs):
        super().update(context, arm, reward)
        self.total_steps += 1
        self.alpha = self.first_alpha / np.log(self.total_steps + 2)

    def get_name(self):
        return f"AdaptiveLinUCB(alpha={self.first_alpha})"


# --- LinUCB with decaying alpha ---
class DecayingAlphaLinUCB(LinUCB):
    def __init__(self, n_arms: int, dim: int, alpha: float = 1, beta=0.99, *args, **kwargs):
        super().__init__(n_arms, dim, alpha)
        self.first_alpha = self.alpha
        self.beta = beta
        self.total_steps = 0

    def update(self, context, arm, reward, *args, **kwargs):
        super().update(context, arm, reward)
        self.alpha *= self.beta

    def get_name(self):
        return f"DecayingAlphaLinUCB(alpha={self.first_alpha}, beta={self.beta})"
