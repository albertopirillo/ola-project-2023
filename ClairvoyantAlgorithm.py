import numpy as np
import logging
from Environment import Environment


class ClairvoyantAlgorithm:
    def __init__(self, environment: Environment) -> None:
        self.environment = environment
        self.optimal_conv_rates: list[float] = []
        self.optimal_prices: list[float] = []
        self.optimal_prices_id: list[int] = []
        self.optimal_bids: list[float] = []
        self.optimal_bids_id: list[int] = []
        self.optimal_rewards: list[float] = []
        self.compute_optimal_solution()

    def compute_optimal_solution(self) -> None:
        for user_class in range(self.environment.num_classes):
            # Find the best price which requires considering both the conversion rate, and the actual price
            best_arm_id: int = int(np.argmax(self.environment.arms_mean[user_class]))
            best_conv_rate: float = self.environment.conv_rates[user_class][best_arm_id]
            best_price: float = self.environment.prices[best_arm_id]

            # Optimize the reward w.r.t. the bid for each class independently
            rewards: list[float] = [self.environment.compute_reward(best_arm_id, bid_id, user_class)
                                    for bid_id in range(len(self.environment.bids))]

            # Update cached values
            self.optimal_conv_rates.append(best_conv_rate)
            self.optimal_prices.append(best_price)
            self.optimal_prices_id.append(best_arm_id)
            self.optimal_bids.append(self.environment.bids[np.argmax(rewards)])
            self.optimal_bids_id.append(int(np.argmax(rewards)))
            self.optimal_rewards.append(np.max(rewards))

            logging.debug('Clairvoyant algorithm')
            logging.debug(f'Class {user_class}')
            logging.debug(f'Optimal conversion rate: {best_conv_rate}')
            logging.debug(f'Optimal price: {best_price}')
            logging.debug(f'Optimal bid: {np.argmax(rewards)}')
            logging.debug(f'Optimal reward: {np.max(rewards)}')
