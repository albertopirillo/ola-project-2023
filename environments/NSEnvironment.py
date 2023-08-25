import numpy as np
from environments.Environment import Environment


class NSEnvironment(Environment):
    def __init__(self, num_classes: int, bids: np.ndarray[float], prices: np.ndarray[float], noise_mean: float,
                 noise_std: float, conv_rates: np.ndarray[float], class_probabilities: np.ndarray[float],
                 horizon: int, phase_length: int) -> None:
        super().__init__(num_classes, bids, prices, noise_mean, noise_std, conv_rates, class_probabilities)
        self.horizon = horizon
        self.phase_length = phase_length
        self.n_phases = self.conv_rates.shape[0]
        self.phases_size = self.horizon / self.n_phases

    def round(self, pulled_arm, t):
        extracted_class: int = np.random.choice(self.num_classes, p=self.class_probabilities)
        current_phase = (t // self.phase_length) % self.n_phases
        features: np.ndarray[int] = np.zeros(2)

        match extracted_class:
            case 0:
                features[0] = 0
                features[1] = 0
            case 1:
                features[0] = np.random.choice(np.arange(2), size=1)
                features[1] = 1 - features[0]
            case 2:
                features[0] = 1
                features[1] = 1

        # minmax scale may compute a rounding error giving an arms_mean value
        # slightly higher than 1 which causes an error when computing reward
        if self.arms_mean[current_phase][extracted_class][pulled_arm] >= 1.:
            self.arms_mean[current_phase][extracted_class][pulled_arm] = 1
        reward: int = np.random.binomial(1, self.arms_mean[current_phase][extracted_class][pulled_arm])
        return reward

    def compute_reward(self, price_index: int, bid_index: int, user_class: int, phase: int) -> float:
        """
        Compute the reward using the true values from the environment.
        :param price_index: the index of the chosen price
        :param bid_index: the index of the chosen bid
        :param user_class: the class of the current user
        :param phase: the current phase of the non-stationary environment
        :return:
        """
        conv_rate = self.conv_rates[phase][user_class][price_index]
        price = self.prices[price_index]
        cost = 0.0
        bid = self.bids[bid_index]
        daily_clicks = self.bid_to_clicks(bid, user_class)
        daily_cost = self.bid_to_daily_cost(bid, user_class)
        return super().actual_reward(conv_rate, price, cost, daily_clicks, daily_cost)
