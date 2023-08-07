import numpy as np
import logging
from Environment import Environment
from utils import compute_reward_clairvoyant


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
            rewards: list[float] = [compute_reward_clairvoyant(self.environment, best_conv_rate, best_price, bid, user_class)
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
