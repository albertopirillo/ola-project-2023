from learners.Learner import Learner
import numpy as np


class TSLearner(Learner):
    def __init__(self, n_arms: int) -> None:
        super().__init__(n_arms)
        self.beta_parameters: np.ndarray[float] = np.ones((n_arms, 2))

    def pull_arm(self) -> int:  # in TS we pull every arm, and then we select the arm with higher value of reward
        idx: int = int(np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])))
        return idx

    def update(self, pulled_arm: int, reward: int) -> None:
        self.t += 1  # update the number of rounds
        self.update_observations(pulled_arm, reward)  # update the observation of the pulled arm
        # the first beta parameter counts the number of successes
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward

    def get_empirical_means(self) -> np.ndarray[float]:
        return self.beta_parameters[:, 0] / (self.beta_parameters[:, 0] + self.beta_parameters[:, 1])
