import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

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
T = 150
n_experiments = 15

# History
instantaneous_reward_clairvoyant = np.zeros(shape=(n_experiments, T))
instantaneous_reward_ucb1 = np.zeros(shape=(n_experiments, T))
instantaneous_reward_ts = np.zeros(shape=(n_experiments, T))

instantaneous_regret_clairvoyant = np.zeros(shape=(n_experiments, T))
instantaneous_regret_ucb1 = np.zeros(shape=(n_experiments, T))
instantaneous_regret_ts = np.zeros(shape=(n_experiments, T))

if __name__ == '__main__':
    for e in trange(n_experiments):
        # For every experiment, we define new environment and learners
        env = Environment(num_classes, bids, prices, noise_mean, noise_std, arms_mean, class_probabilities)
        # No information on pricing nor advertising
        clairvoyant = ClairvoyantAlgorithm(env)
        opt_reward = clairvoyant.optimal_rewards[0]
        # Learners
        ucb1_learner = UCB1Learner(num_arms_pricing)
        ts_learner = TSLearner(num_arms_pricing)
        gp_ucb_learner = GPUCBLearner(num_arms_advertising, bids)
        gp_ts_learner = GPTSLearner(num_arms_advertising, bids)

        for t in trange(T):
            # Clairvoyant algorithm
            instantaneous_reward_clairvoyant[e][t] = opt_reward
            instantaneous_regret_clairvoyant[e][t] = 0

            # UCB1 and GP-UCB learners
            pulled_arm = ucb1_learner.pull_arm()
            pricing_reward = env.round(pulled_arm)
            ucb1_learner.update(pulled_arm, pricing_reward)

            prices_dot_conv = ucb1_learner.empirical_means * prices
            best_arm_id = int(np.argmax(prices_dot_conv))

            pulled_arm = gp_ucb_learner.pull_arm()
            total_reward = env.compute_reward(best_arm_id, pulled_arm, user_class=0)
            gp_ucb_learner.update(pulled_arm, total_reward)

            instantaneous_reward_ucb1[e][t] = total_reward
            regret = opt_reward - total_reward
            instantaneous_regret_ucb1[e][t] = regret

            # TS and GP-TS learners
            pulled_arm = ts_learner.pull_arm()
            pricing_reward = env.round(pulled_arm)
            ts_learner.update(pulled_arm, pricing_reward)

            prices_dot_conv = ts_learner.get_empirical_means() * prices
            best_arm_id = int(np.argmax(prices_dot_conv))

            pulled_arm = gp_ts_learner.pull_arm()
            total_reward = env.compute_reward(best_arm_id, pulled_arm, user_class=0)
            gp_ts_learner.update(pulled_arm, total_reward)

            instantaneous_reward_ts[e][t] = total_reward
            regret = opt_reward - total_reward
            instantaneous_regret_ts[e][t] = regret

    plot_statistics(instantaneous_reward_clairvoyant, instantaneous_regret_clairvoyant, 'Clairvoyant', 'Step 3')
    plot_statistics(instantaneous_reward_ucb1, instantaneous_regret_ucb1, 'UCB1 & GP-UCB', 'Step 3')
    plot_statistics(instantaneous_reward_ts, instantaneous_regret_ts, 'TS & GP-TS', 'Step 3')
    plt.tight_layout()
    plt.show()
