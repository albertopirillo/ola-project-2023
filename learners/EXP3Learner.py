import numpy as np
from learners.Learner import Learner


class EXP3Learner(Learner):
    def __init__(self, n_arms: int, gamma: float) -> None:
        super().__init__(n_arms)
        self.weights = np.ones(n_arms)
        self.estimated_reward = np.zeros(n_arms)
        self.gamma = gamma
        self.probabilities = np.zeros(n_arms)

    def draw(self) -> int:
        return np.random.choice(np.arange(self.n_arms), size=1, p=self.probabilities)

    def pull_arm(self) -> int:
        weight_sum = float(sum(self.weights))
        for w in range(len(self.weights)):
            self.probabilities[w] = ((1.0 - self.gamma) * (self.weights[w] / weight_sum) + (self.gamma / self.n_arms))
        index = self.draw()
        return index

    def update(self, pulled_arm: int, reward: float) -> None:
        self.t += 1
        self.estimated_reward[pulled_arm] = reward / self.probabilities[pulled_arm]
        self.weights[pulled_arm] *= np.exp(self.estimated_reward[pulled_arm] * self.gamma / self.n_arms)
