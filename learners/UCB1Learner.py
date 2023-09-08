import numpy as np

from learners.Learner import Learner


class UCB1Learner(Learner):
    def __init__(self, n_arms: int) -> None:
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf] * n_arms)
        # Hyperparameter that controls the exploration-exploitation tradeoff.
        # Higher values encourage more exploration.
        # Default is 1.0
        self.beta = 1.0

    def pull_arm(self, prices: np.ndarray[float] = None) -> int:
        upper_conf = self.empirical_means + self.beta * self.confidence
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])

    def update(self, pull_arm: int, reward: float) -> None:
        # No assumptions on the distribution of the reward
        self.t += 1
        self.empirical_means[pull_arm] = (self.empirical_means[pull_arm] * (self.t - 1) + reward) / self.t
        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = ((2 * np.log(self.t) / n_samples) ** 0.5) if n_samples > 0 else np.inf
        self.update_observations(pull_arm, reward)

    def get_best_expected_value(self) -> float:
        best = np.max(self.empirical_means)
        return best
