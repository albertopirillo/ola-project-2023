import numpy as np


class Environment:
    def __init__(self, bids, cumulative_daily_cost, prices, noises_mean, noises_std, arms_mean, class_probabilities):
        self.bids = bids                                #array containing the bids
        self.daily_budgets = cumulative_daily_cost              #
        self.prices = prices                            #prices of the tickets
        self.noises_mean = noises_mean                  #mean of the noise to add to the drawn sample
        self.noises_std = noises_std                    #std deviatin of the noise to add to the drawn sample
        self.arms_mean = arms_mean                      #bernoulli distribution of the arms for the prices
        self.class_probabilities = class_probabilities  #distributions of the classes

    def round(self, arm_index):
        extracted_class = np.random.choice(np.arange(3), size=1, p=self.class_probabilities)
        features = np.zeros(2)
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

        reward = np.random.binomial(1, self.arms_mean[arm_index], size=1)
        return reward


# Curves expressing the average dependence between the bid and the number of clicks
def bid_to_clicks(bids, user_class):
    if user_class == 0:
        return 100 * (1.0 - np.exp(-4 * bids + 3 * bids ** 3))
    if user_class == 1:
        return 100 * (1.0 - np.exp(-5 * bids + 3 * bids ** 2))
    if user_class == 2:
        return 100 * (1.0 - np.exp(-6 * bids + 3 * bids ** 2))


# Curves expressing the average dependence between the bid and the cumulative daily cost
def bid_to_daily_cost(bids, user_class):
    if user_class == 0:
        return 100 * (1.0 - np.exp(-4 * bids + 5 * bids ** 2))
    if user_class == 1:
        return 100 * (1.0 - np.exp(-5 * bids + 6 * bids ** 3))
    if user_class == 2:
        return 100 * (1.0 - np.exp(-6 * bids + 7 * bids ** 3))

# Function to add noise to the bid_to_click curve whenever a sample is drawn
def generate_observation_from_click(bids, user_class, noise_mean, noise_std):
    return bid_to_clicks(bids, user_class) + np.random.normal(noise_mean, noise_std)

# Function to add noise to the bid_to_daily_cost curve whenever a sample is drawn
def generate_observation_from_daily_cost(bids, user_class, noise_mean, noise_std):
    return bid_to_daily_cost(bids, user_class) + np.random.normal(noise_mean, noise_std)
