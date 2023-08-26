import numpy as np
import logging

from environments.ClairvoyantAlgorithm import ClairvoyantAlgorithm
from environments.NSEnvironment import NSEnvironment


class NSClairvoyantAlgorithm(ClairvoyantAlgorithm):
    def __init__(self, environment: NSEnvironment) -> None:
        self.environment = environment
        super().__init__(environment)

    def compute_optimal_solution(self) -> None:
        for user_class in range(self.environment.num_classes):
            for phase in range(self.environment.n_phases):
                # Find the best price which requires considering both the conversion rate, and the actual price
                best_arm_id: int = int(np.argmax(self.environment.conv_rates[phase][user_class]))
                best_conv_rate: float = self.environment.conv_rates[phase][user_class][best_arm_id]
                best_price: float = self.environment.prices[best_arm_id]

                # Optimize the reward w.r.t. the bid for each class independently
                rewards: list[float] = [self.environment.compute_reward(best_arm_id, bid_id, user_class, phase)
                                        for bid_id in range(len(self.environment.bids))]

                # Update cached values
                self.optimal_conv_rates.append(best_conv_rate)
                self.optimal_prices.append(best_price)
                self.optimal_prices_id.append(best_arm_id)
                self.optimal_bids.append(self.environment.bids[np.argmax(rewards)])
                self.optimal_bids_id.append(int(np.argmax(rewards)))
                self.optimal_rewards.append(np.max(rewards))

                logging.debug('Clairvoyant algorithm')
                logging.debug(f'Phase {phase + 1}')
                logging.debug(f'Class {user_class}')
                logging.debug(f'Optimal conversion rate: {best_conv_rate}')
                logging.debug(f'Optimal price: {best_price}')
                logging.debug(f'Optimal bid: {np.argmax(rewards)}')
                logging.debug(f'Optimal reward: {np.max(rewards)}')
