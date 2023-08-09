import json
import numpy as np
from typing import Self
from sklearn.preprocessing import minmax_scale


class Environment:
    def __init__(self, num_classes: int, bids: np.ndarray[float], prices: np.ndarray[float], noise_mean: float,
                 noise_std: float, arms_mean: np.ndarray[float], class_probabilities: np.ndarray[float]) -> None:
        self.num_classes = num_classes  # number of classes
        self.bids = bids  # array containing the bids
        self.prices = prices  # prices of the tickets
        self.noise_mean = noise_mean  # mean of the noise to add to the drawn sample
        self.noise_std = noise_std  # std deviation of the noise to add to the drawn sample
        self.arms_mean = arms_mean  # matrix containing bernoulli distributions of the arms for the prices
        self.class_probabilities = class_probabilities  # distributions of the classes
        # When learning, consider the [0,1]-normalized conv_rates * prices
        self.mean_per_prices = minmax_scale((self.arms_mean * self.prices), axis=1)

    @classmethod
    def from_json(cls, json_path: str) -> Self:
        with open(json_path) as f:
            data = json.load(f)
        # Convert from JSON arrays to Numpy arrays
        data['bids'] = np.linspace(*data['bids'])
        data['prices'] = np.array(data['prices'])
        data['arms_mean'] = np.array(data['arms_mean'])
        data['class_probabilities'] = np.array(data['class_probabilities'])
        return cls(**data)

    def round(self, arm_index) -> int:
        extracted_class: int = np.random.choice(self.num_classes, p=self.class_probabilities)
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

        reward: int = np.random.binomial(1, self.mean_per_prices[extracted_class][arm_index])
        return reward

    # Function to add noise to the bid_to_click curve whenever a sample is drawn
    def generate_observation_from_click(self, bids: np.ndarray[float], user_class: int) -> float:
        size = 1 if isinstance(bids, float) else len(bids)
        clicks = self.bid_to_clicks(bids, user_class) + np.random.normal(self.noise_mean, self.noise_std, size=size)
        return np.maximum(clicks, 0)

    # Function to add noise to the bid_to_daily_cost curve whenever a sample is drawn
    def generate_observation_from_daily_cost(self, bids: np.ndarray[float], user_class: int) -> float:
        size = 1 if isinstance(bids, float) else len(bids)
        cost = self.bid_to_daily_cost(bids, user_class) + np.random.normal(self.noise_mean, self.noise_std, size=size)
        return np.maximum(cost, 0)

    @staticmethod
    def bid_to_clicks(bids: np.ndarray[float], user_class: int) -> float:
        """ Curves expressing the average dependence between the bid and the number of clicks """
        if user_class == 0:
            clicks = 100 * (1.0 - np.exp(-4 * bids + 3 * bids ** 3))
        elif user_class == 1:
            clicks = 100 * (1.0 - np.exp(-5 * bids + 3 * bids ** 2))
        else:
            clicks = 100 * (1.0 - np.exp(-6 * bids + 3 * bids ** 2))
        return np.maximum(clicks, 0)

    @staticmethod
    def bid_to_daily_cost(bids: np.ndarray[float], user_class: int):
        """ Curves expressing the average dependence between the bid and the cumulative daily cost """
        if user_class == 0:
            cost = 100 * (5.5 - np.exp(-7 * bids + 5 * bids ** 2))
        elif user_class == 1:
            cost = 50 * (3.5 - np.exp(-10 * bids + 6 * bids ** 3))
        else:
            cost = 75 * (4.0 - np.exp(-8 * bids + 3 * bids ** 3))
        return np.maximum(cost, 0)

    def compute_reward(self, price_index: int, bid_index: int, user_class: int) -> float:
        """
        Compute the reward using the true values from the environment.
        :param price_index: the index of the chosen price
        :param bid_index: the index of the chosen bid
        :param user_class: the class of the current user
        :return:
        """
        conv_rate = self.arms_mean[user_class][price_index]
        price = self.prices[price_index]
        bid = self.bids[bid_index]
        daily_clicks = self.bid_to_clicks(bid, user_class)
        daily_cost = self.bid_to_daily_cost(bid, user_class)
        return self.actual_reward(conv_rate, price, bid, daily_clicks, daily_cost)

    @staticmethod
    def actual_reward(conv_rate: float, price: float, bid: float, daily_clicks: float, daily_cost: float) -> float:
        """
        Performs the actual computation of the reward given the parameters passed by the other functions.
        The reward is given by: (daily_clicks * conv_rate * margin) - cumulative_daily_cost
        The margin is given by the price minus the bid.
        This method should never be called directly.
        :param conv_rate: the conversion rate
        :param price: the price
        :param bid: the bid
        :param daily_clicks: the number of daily clicks
        :param daily_cost: the daily cost
        :return:
        """
        return (daily_clicks * conv_rate * (price - bid)) - daily_cost
