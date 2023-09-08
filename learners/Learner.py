from abc import abstractmethod

import numpy as np


class Learner:
    def __init__(self, n_arms: int) -> None:
        self.n_arms = n_arms
        self.t: int = 0  # round value
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm: int, reward: int | float) -> None:
        """ Update the observations once the rewards are returned by the environment """
        self.rewards_per_arm[pulled_arm].append(reward)  # update the reward of the arm pulled by the learner
        self.collected_rewards = np.append(self.collected_rewards, reward)

    @abstractmethod
    def pull_arm(self, prices: np.ndarray[float] = None) -> int:
        pass

    @abstractmethod
    def get_best_expected_value(self) -> float:
        pass

    @abstractmethod
    def update(self, pull_arm: int, reward: float) -> None:
        pass
