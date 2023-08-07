import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

# import warnings
# from sklearn.exceptions import ConvergenceWarning
# warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# import logging
# logging.basicConfig(level=logging.DEBUG)

from ClairvoyantAlgorithm import ClairvoyantAlgorithm
from Environment import Environment
from GPTSLearner import GPTSLearner
from GPUCBLearner import GPUCBLearner
from TSLearner import TSLearner
from UCB1Learner import UCB1Learner
from utils import plot_statistics

# Environment parameters
num_classes = 3
num_arms_pricing = 5
num_arms_advertising = 100
bids = np.linspace(0, 1, 100)
prices = np.array([15.5, 30.7, 60.2, 70.6, 90.8])
noise_mean = 0.0
noise_std = 5.0
arms_mean = np.array([[0.4, 0.7, 0.3, 0.2, 0.1],
                      [0.1, 0.2, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.3, 0.1, 0.0]])
class_probabilities = np.array([1, 0, 0])

# Simulation parameters
T = 125
n_experiments = 5

# History
instantaneous_reward_clairvoyant = np.zeros(shape=(n_experiments, T))
instantaneous_reward_ucb1 = np.zeros(shape=(n_experiments, T))
instantaneous_reward_ts = np.zeros(shape=(n_experiments, T))

instantaneous_regret_clairvoyant = np.zeros(shape=(n_experiments, T))
instantaneous_regret_ucb1 = np.zeros(shape=(n_experiments, T))
instantaneous_regret_ts = np.zeros(shape=(n_experiments, T))


def compute_reward(conv_rate: float, price: float, bid: float) -> float:
    return (env.generate_observation_from_click(bid, user_class=0) * conv_rate * (price - bid)) - env.generate_observation_from_daily_cost(bid, user_class=0)


if __name__ == '__main__':
    for e in trange(n_experiments):
        # For every experiment, we define new environment and learners
        env = Environment(num_classes, bids, prices, noise_mean, noise_std, arms_mean, class_probabilities)
        observations = env.generate_observation_from_click(bids, user_class=0)
        clairvoyant = ClairvoyantAlgorithm(env)
        ucb1_learner = UCB1Learner(num_arms_pricing)
        ts_learner = TSLearner(num_arms_pricing)
        gp_ucb_learner = GPUCBLearner(num_arms_advertising, observations)
        gp_ts_learner = GPTSLearner(num_arms_advertising, observations)


        for t in range(T):
            # Clairvoyant Algorithm
            opt_reward = clairvoyant.compute_optimal_solution()

            instantaneous_reward_clairvoyant[e][t] = opt_reward
            instantaneous_regret_clairvoyant[e][t] = 0

            print(f'Clairvoyant')
            print(f'Optimal reward: {opt_reward}')

            # UCB1 and GP-UCB Learner
            pulled_arm = ucb1_learner.pull_arm()
            pricing_reward = env.round(pulled_arm)
            ucb1_learner.update(pulled_arm, pricing_reward)

            # TODO: check that empirical means are correct
            prices_dot_conv = ucb1_learner.empirical_means * prices
            best_arm_id = int(np.argmax(prices_dot_conv))
            best_conv_rate = ucb1_learner.empirical_means[best_arm_id]
            best_price = prices[best_arm_id]


            pulled_arm = gp_ucb_learner.pull_arm()
            reward = compute_reward(best_conv_rate, best_price, bids[pulled_arm])
            gp_ucb_learner.update(pulled_arm, reward)

            instantaneous_reward_ucb1[e][t] = reward[0]
            regret = opt_reward - reward[0]
            instantaneous_regret_ucb1[e][t] = regret

            print(f'UCB')
            print(f'Best pricing arm id: {best_arm_id}')
            print(f'Best conv rate: {best_conv_rate}')
            print(f'Best price: {best_price}')
            print(f'Best advertising arm id: {pulled_arm}')
            print(f'Best bid: {bids[pulled_arm]}')
            print(f'Reward: {reward}')


            # TS and GP Thompson Sampling Learner
            pulled_arm = ts_learner.pull_arm()
            pricing_reward = env.round(pulled_arm)
            ts_learner.update(pulled_arm, pricing_reward)


            # alpha / (alpha + beta)
            # TODO: check that empirical means are correct
            ts_empirical_means = ts_learner.get_empirical_means()
            prices_dot_conv = ts_empirical_means * prices
            best_arm_id = int(np.argmax(prices_dot_conv))
            best_conv_rate = ts_empirical_means[best_arm_id]
            best_price = prices[best_arm_id]

            pulled_arm = gp_ts_learner.pull_arm()
            reward = compute_reward(best_conv_rate, best_price, bids[pulled_arm])
            gp_ts_learner.update(pulled_arm, reward)

            print(f'TS')
            print(f'Best pricing arm id: {best_arm_id}')
            print(f'Best conv rate: {best_conv_rate}')
            print(f'Best price: {best_price}')
            print(f'Best advertising arm id: {pulled_arm}')
            print(f'Best bid: {bids[pulled_arm]}')
            print(f'Reward: {reward}')
            print('------------------------------------------------------------------')

            instantaneous_reward_ucb1[e][t] = reward[0]
            regret = opt_reward - reward[0]
            instantaneous_regret_ucb1[e][t] = regret


    plot_statistics(instantaneous_reward_clairvoyant, instantaneous_regret_clairvoyant, 'Clairvoyant', 'Step 3')
    plot_statistics(instantaneous_reward_ucb1, instantaneous_regret_ucb1, 'UCB1 & GP-UCB', 'Step 3')
    plot_statistics(instantaneous_reward_ts, instantaneous_regret_ts, 'TS & GP-TS', 'Step 3')
    plt.tight_layout()
    plt.show()
