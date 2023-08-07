from Learner import Learner
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPUCBLearner(Learner):
    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10

        self.nt = [0.0001 for _ in range(0, self.n_arms)]
        self.beta = [10.0 for _ in range(0, self.n_arms)]

        self.pulled_arms = []
        alpha = 0.5  # standard deviation of the noise
        kernel = C(1., (1e-12, 1e3)) * RBF(1., (1e-12, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=True, n_restarts_optimizer=5)

    def update_observations(self, arm_idx, reward):
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T  # input for training: the pulled arms
        y = self.collected_rewards  # target of the model
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update_betas(self):
        for i in range(0, len(self.arms)):
            self.beta[i] = np.sqrt((2 * np.log(self.t)) / self.nt[i])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()
        self.nt[pulled_arm] += 1
        self.update_betas()

    def pull_arm(self):
        sampled_value = np.argmax(self.means + self.sigmas * self.beta)
        return sampled_value
