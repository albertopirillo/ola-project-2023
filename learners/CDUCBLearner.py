from learners.UCB1Learner import UCB1Learner
from learners.CUSUM import CUSUM
import numpy as np


class CDUCBLearner(UCB1Learner):
    def __init__(self, n_arms: int, M: int = 100, eps: float = 0.05, h: int = 20, alpha: float = 0.01) -> None:
        super().__init__(n_arms)
        self.change_detection = [CUSUM(M, eps, h) for _ in range(n_arms)]
        self.valid_rewards_per_arms = [[] for _ in range(n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.alpha = alpha
        # Hyperparameter that controls the exploration-exploitation tradeoff
        self.beta = 3.0

    def pull_arm(self) -> int:
        # with probability 1-alpha select the arm with the highest upper confidence bound
        if np.random.binomial(1, 1 - self.alpha):
            upper_conf = self.empirical_means + self.beta * self.confidence
            upper_conf[np.isinf(upper_conf)] = 1e3
            return np.random.choice(np.where(upper_conf == upper_conf.max())[0])
        else:
            return np.random.choice(self.n_arms)  # with probability alpha pull a random arm

    def update(self, pulled_arm: int, reward: float) -> None:
        self.t += 1
        if self.change_detection[pulled_arm].update(reward):
            self.detections[pulled_arm].append(self.t)
            self.valid_rewards_per_arms[pulled_arm] = []
            self.change_detection[pulled_arm].reset()
        self.update_observations(pulled_arm, reward)
        self.empirical_means[pulled_arm] = np.mean(self.valid_rewards_per_arms[pulled_arm])
        total_valid_samples = sum([len(x) for x in self.valid_rewards_per_arms])
        for a in range(self.n_arms):
            n_samples = len(self.valid_rewards_per_arms[a])
            self.confidence[a] = (2 * np.log(total_valid_samples) / n_samples) ** 0.5 if n_samples > 0 else np.inf

    def update_observations(self, pulled_arm: int, reward: float) -> None:
        self.rewards_per_arm[pulled_arm].append(reward)
        self.valid_rewards_per_arms[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
