from learners.UCB1Learner import UCB1Learner
import numpy as np


class SWUCB1Learner(UCB1Learner):
    def __init__(self, n_arms: int, windows_size: int) -> None:
        super().__init__(n_arms)
        self.windows_size = windows_size
        self.pulled_arms = np.array([])
        # Hyperparameter that controls the exploration-exploitation tradeoff
        self.beta = 90.8

    def update(self, pulled_arm: int, reward: float) -> None:
        self.t += 1
        self.pulled_arms = np.append(self.pulled_arms, pulled_arm)
        n_pulled_arm = np.sum(self.pulled_arms[-self.windows_size:] == pulled_arm)
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (
                    n_pulled_arm - 1) + reward) / n_pulled_arm
        for arm in range(self.n_arms):
            n_samples = np.sum(self.pulled_arms[-self.windows_size:] == arm)
            self.confidence[arm] = (2 * np.log(
                np.min((self.t, self.windows_size), axis=0)) / n_samples) ** 0.5 if n_samples > 0 else np.inf
        self.update_observations(pulled_arm, reward)
