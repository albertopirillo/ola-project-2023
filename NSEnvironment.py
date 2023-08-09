from Environment import Enivornment
import numpy as np

class NS_Environment(Enivornment):
    def __init__(self, n_arms, probabilities, horizon):
        super().__init__(n_arms, probabilities)
       #self.t = 0
        n_phases = len(self.probabilities)
        self.phases_size = horizon / n_phases

    def round(self, pulled_arm):
        current_phase = int(self.t / self.phases_size)
        p = self.probabilities[current_phase][pulled_arm]
        reward = np.random.binomial(1, p)
        #self.t += 1
        return reward