import json
from Environment import Environment
import numpy as np
from sklearn.preprocessing import minmax_scale

class NSEnvironment(Environment):
    def __init__(self, num_classes: int, bids: np.ndarray[float], prices: np.ndarray[float], noise_mean: float,
            noise_std: float, conv_rates: np.ndarray[float], class_probabilities: np.ndarray[float],
            phases_probabilities: np.ndarray[float], horizon: int) -> None:
        super().__init__(num_classes, bids, prices, noise_mean, noise_std, conv_rates, class_probabilities)
        self.phases_probabilities = phases_probabilities
        self.horizon = horizon
        n_phases = len(self.phases_probabilities)
        self.phases_size = self.horizon / n_phases
        self.arms_mean = [minmax_scale((self.phases_probabilities[i] * self.prices), axis=1) for i in range(3)]
    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path) as f:
            data = json.load(f)
        # Convert from JSON arrays to Numpy arrays
        data['bids'] = np.linspace(*data['bids'])
        data['prices'] = np.array(data['prices'])
        data['conv_rates'] = np.array(data['conv_rates'])
        data['class_probabilities'] = np.array(data['class_probabilities'])
        data['phases_probabilities'] = np.array(data['phases_probabilities'])
        return cls(**data)

    def round(self, pulled_arm, t):
        extracted_class: int = np.random.choice(self.num_classes, p=self.class_probabilities)
        current_phase = int(t / self.phases_size)
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
        print('phase: ',current_phase)
        print(f'arms_mean:{self.arms_mean[extracted_class][current_phase][pulled_arm]}')
        # minmax scale may compute a rounding error giving an arms_mean value slightly higher than 1 which causes an error when computing reward
        if self.arms_mean[extracted_class][current_phase][pulled_arm] >= 1.:
            self.arms_mean[extracted_class][current_phase][pulled_arm] = 1
        reward: int = np.random.binomial(1, self.arms_mean[extracted_class][current_phase][pulled_arm])
        return reward
    
    def compute_reward(self, price_index: int, bid_index: int, user_class: int, phase: int) -> float:
        """
        Compute the reward using the true values from the environment.
        :param price_index: the index of the chosen price
        :param bid_index: the index of the chosen bid
        :param user_class: the class of the current user
        :return:
        """
        conv_rate = self.phases_probabilities[user_class][phase][price_index]
        price = self.prices[price_index]
        cost = 0.0
        bid = self.bids[bid_index]
        daily_clicks = self.bid_to_clicks(bid, user_class)
        daily_cost = self.bid_to_daily_cost(bid, user_class)
        return self.actual_reward(conv_rate, price, cost, daily_clicks, daily_cost)

    @staticmethod
    def actual_reward(conv_rate: float, price: float, cost: float, daily_clicks: float, daily_cost: float) -> float:
        """
        Performs the actual computation of the reward given the parameters passed by the other functions.
        The reward is given by: (daily_clicks * conv_rate * margin) - cumulative_daily_cost
        The margin is given by the price minus the cost.
        This method should never be called directly.
        :param conv_rate: the conversion rate of the product
        :param price: the price of the product
        :param cost: the cost to the company of the product
        :param bid: the bid
        :param daily_clicks: the number of daily clicks
        :param daily_cost: the cumulative daily cost due to advertising
        :return:
        """
        return (daily_clicks * conv_rate * (price - cost)) - daily_cost