import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from ClairvoyantAlgorithm import ClairvoyantAlgorithm
from Environment import Environment
from TSLearner import TSLearner
from UCB1Learner import UCB1Learner
from utils import plot_statistics

# Simulation parameters
T = 365
n_experiments = 1000

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
        env = Environment.from_json('data/environment.json')
        # Advertising curves (optimal bid) are known
        clairvoyant = ClairvoyantAlgorithm(env)
        opt_reward = clairvoyant.optimal_rewards[0]
        opt_bid_id = clairvoyant.optimal_bids_id[0]
        # Learners
        ucb1_learner = UCB1Learner(len(env.prices))
        ts_learner = TSLearner(len(env.prices))

        for t in range(T):
            # Clairvoyant algorithm
            instantaneous_reward_clairvoyant[e][t] = opt_reward
            instantaneous_regret_clairvoyant[e][t] = 0

            # UCB1 learner
            pulled_arm = ucb1_learner.pull_arm()
            pricing_reward = env.round(pulled_arm)
            ucb1_learner.update(pulled_arm, pricing_reward)

            total_reward = env.compute_reward(pulled_arm, opt_bid_id, user_class=0)
            instantaneous_reward_ucb1[e][t] = total_reward
            regret = opt_reward - total_reward
            instantaneous_regret_ucb1[e][t] = regret

            # Thompson Sampling learner
            pulled_arm = ts_learner.pull_arm()
            pricing_reward = env.round(pulled_arm)
            ts_learner.update(pulled_arm, pricing_reward)

            total_reward = env.compute_reward(pulled_arm, opt_bid_id, user_class=0)
            instantaneous_reward_ts[e][t] = total_reward
            regret = opt_reward - total_reward
            instantaneous_regret_ts[e][t] = regret

    plot_statistics(instantaneous_reward_clairvoyant, instantaneous_regret_clairvoyant, 'Clairvoyant', 'Step 1')
    plot_statistics(instantaneous_reward_ucb1, instantaneous_regret_ucb1, 'UCB1', 'Step 1')
    plot_statistics(instantaneous_reward_ts, instantaneous_regret_ts, 'TS', 'Step 1')
    plt.tight_layout()
    plt.show()
