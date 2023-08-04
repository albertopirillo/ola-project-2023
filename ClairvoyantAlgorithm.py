import numpy as np
from Environment import Environment


class ClairvoyantAlgorithm:
    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    def compute_optimal_solution(self) -> float:
        total_reward: float = 0
        # Find the best price for each class, which requires considering both the conversion rate and the price
        # Arm's mean is the conversion rate
        prices_dot_conv: np.ndarray[float] = self.environment.arms_mean * self.environment.prices
        best_arms_ids: np.ndarray[int] = np.argmax(prices_dot_conv, axis=1)
        best_arms_mean: np.ndarray[float] = np.max(prices_dot_conv, axis=1)

        # Optimize the reward w.r.t. the bid for each class independently
        for user_class in range(self.environment.num_classes):
            rewards: list[float] = [self.compute_reward(bid, user_class, best_arms_ids, best_arms_mean) for bid in
                                    self.environment.bids]
            total_reward += np.max(rewards)

        # Return the reward of the optimal solution
        return total_reward

    def compute_reward(self, bid: float, user_class: int, best_prices_ids: np.ndarray[int],
                       best_arms_mean: np.ndarray[float]) -> float:
        """ Compute the reward = (number of daily clicks * conversion rate * margin) - the cumulative daily costs """
        return ((self.environment.bid_to_clicks(bid, user_class) * best_arms_mean[user_class]
                * self.environment.prices[best_prices_ids[user_class]] - bid) -
                self.environment.bid_to_daily_cost(bid, user_class))
        # * self.environment.bid_to_clicks(bid, user_class))
