import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from learners.Learner import Learner


class GPTSLearner(Learner):
    def __init__(self, n_arms: int, arms: np.ndarray[float]) -> None:
        super().__init__(n_arms)
        self.arms = arms
        self.means: np.ndarray[float] = np.zeros(self.n_arms)
        self.sigmas: np.ndarray[float] = np.ones(self.n_arms) * 5.0
        self.pulled_arms: list[float] = []

        # GP parameters
        # length_scale: how much a prediction is influenced by surrounding points.
        # Large values of the length_scale mean very low correlation and result in a very smooth function.
        # alpha: small constant added to prevent numerical issues
        # sigma: controls the noise point, the higher the value, the more noise in the data.
        # Large values of sigma result in a larger confidence interval.
        self.kernel = RBF(length_scale=0.25, length_scale_bounds=(1e-12, 1e5))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.25, normalize_y=True, n_restarts_optimizer=7)

    def update_observations(self, arm_idx: int, reward: float) -> None:
        super().update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self) -> None:
        x = np.atleast_2d(self.pulled_arms).T  # input for training: the pulled arms
        y = self.collected_rewards  # target of the model
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm: int, reward: float) -> None:
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self, prices: np.ndarray[float] = None) -> int:
        sampled_values = np.random.normal(self.means, self.sigmas)
        return int(np.argmax(sampled_values))

    def get_best_expected_value(self) -> float:
        best = np.max(self.means)
        return best
