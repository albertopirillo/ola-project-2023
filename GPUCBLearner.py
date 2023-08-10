from Learner import Learner
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPUCBLearner(Learner):
    def __init__(self, n_arms: int, arms: np.ndarray[float]) -> None:
        super().__init__(n_arms)
        self.arms = arms
        self.means: np.ndarray[float] = np.zeros(self.n_arms)
        self.sigmas: np.ndarray[float] = np.ones(self.n_arms) * 5.0
        self.nt: list[float] = [0.0001 for _ in range(0, self.n_arms)]
        self.beta: list[float] = [10.0 for _ in range(0, self.n_arms)]
        self.pulled_arms = []

        # GP parameters
        self.kernel = C(1., (1e-5, 1e5)) * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.25, normalize_y=True, n_restarts_optimizer=5)

    def update_observations(self, arm_idx: int, reward: float) -> None:
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self) -> None:
        x = np.atleast_2d(self.pulled_arms).T  # input for training: the pulled arms
        y = self.collected_rewards  # target of the model
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update_betas(self) -> None:
        for i in range(0, len(self.arms)):
            self.beta[i] = np.sqrt((2 * np.log(self.t)) / self.nt[i])

    def update(self, pulled_arm: int, reward: float) -> None:
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()
        self.nt[pulled_arm] += 1
        self.update_betas()

    def pull_arm(self) -> int:
        sampled_value = int(np.argmax(self.means + self.sigmas * self.beta))
        return sampled_value
