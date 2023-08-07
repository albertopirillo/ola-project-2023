import numpy as np


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

        reward: int = np.random.binomial(1, self.arms_mean[extracted_class][arm_index])
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

    def bid_to_clicks(self, bids: np.ndarray[float], user_class: int) -> float:
        """ Curves expressing the average dependence between the bid and the number of clicks """
        if user_class == 0:
            clicks = 100 * (1.0 - np.exp(-4 * bids + 3 * bids ** 3))
        elif user_class == 1:
            clicks = 100 * (1.0 - np.exp(-5 * bids + 3 * bids ** 2))
        else:
            clicks = 100 * (1.0 - np.exp(-6 * bids + 3 * bids ** 2))
        return np.maximum(clicks, 0)

    def bid_to_daily_cost(self, bids: np.ndarray[float], user_class: int):
        """ Curves expressing the average dependence between the bid and the cumulative daily cost """
        if user_class == 0:
            cost = 100 * (5.5 - np.exp(-7 * bids + 5 * bids ** 2))
        elif user_class == 1:
            cost = 50 * (3.5 - np.exp(-10 * bids + 6 * bids ** 3))
        else:
            cost = 75 * (4.0 - np.exp(-8 * bids + 3 * bids ** 3))
        return np.maximum(cost, 0)