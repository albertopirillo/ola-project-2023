import numpy as np
import logging
from Environment import Environment


class ClairvoyantAlgorithm:
    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    def compute_optimal_solution(self) -> float:
        class_reward: list[float] = list()

        for user_class in range(self.environment.num_classes):
            # Find the best price which requires considering both the conversion rate, and the actual price
            # Arm's mean is the conversion rate
            prices_dot_conv: np.ndarray[float] = self.environment.arms_mean[user_class] * self.environment.prices
            best_arm_id: int = int(np.argmax(prices_dot_conv))
            best_conv_rate: float = self.environment.arms_mean[user_class][best_arm_id]
            best_price: float = self.environment.prices[best_arm_id]

            # Optimize the reward w.r.t. the bid for each class independently
            rewards: list[float] = [self.compute_reward(bid, user_class, best_price, best_conv_rate)
                                    for bid in self.environment.bids]
            class_reward.append(np.max(rewards))

            logging.debug('Clairvoyant algorithm')
            logging.debug(f'Class {user_class}')
            logging.debug(f'Optimal conversion rate: {best_conv_rate}')
            logging.debug(f'Optimal price: {best_price}')
            logging.debug(f'Optimal bid: {np.argmax(rewards)}')

        # Return the reward of the optimal solution, weighted on the class probabilities
        weighted_reward = float(np.dot(class_reward, self.environment.class_probabilities))
        logging.debug(f'Optimal total reward: {weighted_reward}')
        return weighted_reward

    def compute_reward(self, bid: float, user_class: int, price: float, conv_rate: float) -> float:
        """
         Compute the reward = (number of daily clicks * conversion rate * margin) - the cumulative daily costs
         The margin is given by the price minus the bid
         """
        return (self.environment.bid_to_clicks(bid, user_class) * conv_rate * (
                    price - bid)) - self.environment.bid_to_daily_cost(bid, user_class)
        # * self.environment.bid_to_clicks(bid, user_class))
