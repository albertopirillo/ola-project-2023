import numpy as np


class Learner:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0  # round value
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        """ Update the observations once the rewards are returned by the environment """
        self.rewards_per_arm[pulled_arm].append(reward)  # update the reward of the arm pulled by the learner
        self.collected_rewards = np.append(self.collected_rewards, reward)
