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
        env = Environment.from_json('data/environment.json')
        # No information on pricing nor advertising
        clairvoyant = ClairvoyantAlgorithm(env)
        opt_reward = clairvoyant.optimal_rewards[0]
        # Learners
        ucb1_learner = UCB1Learner(len(env.prices))
        ts_learner = TSLearner(len(env.prices))
        gp_ucb_learner = GPUCBLearner(len(env.bids), env.bids)
        gp_ts_learner = GPTSLearner(len(env.bids), env.bids)

        for t in trange(T):
            # Clairvoyant algorithm
            instantaneous_reward_clairvoyant[e][t] = opt_reward
            instantaneous_regret_clairvoyant[e][t] = 0

            # UCB1 and GP-UCB learners
            pulled_arm_pricing = ucb1_learner.pull_arm()
            pricing_reward = env.round(pulled_arm_pricing)
            ucb1_learner.update(pulled_arm_pricing, pricing_reward)

            pulled_arm_advertising = gp_ucb_learner.pull_arm()
            total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising, user_class=0)
            gp_ucb_learner.update(pulled_arm_advertising, total_reward)

            instantaneous_reward_ucb1[e][t] = total_reward
            regret = opt_reward - total_reward
            instantaneous_regret_ucb1[e][t] = regret

            # TS and GP-TS learners
            pulled_arm_pricing = ts_learner.pull_arm()
            pricing_reward = env.round(pulled_arm_pricing)
            ts_learner.update(pulled_arm_pricing, pricing_reward)

            pulled_arm_advertising = gp_ts_learner.pull_arm()
            total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising, user_class=0)
            gp_ts_learner.update(pulled_arm_advertising, total_reward)

            instantaneous_reward_ts[e][t] = total_reward
            regret = opt_reward - total_reward
            instantaneous_regret_ts[e][t] = regret

    plot_statistics(instantaneous_reward_clairvoyant, instantaneous_regret_clairvoyant, 'Clairvoyant', 'Step 3')
    plot_statistics(instantaneous_reward_ucb1, instantaneous_regret_ucb1, 'UCB1 & GP-UCB', 'Step 3')
    plot_statistics(instantaneous_reward_ts, instantaneous_regret_ts, 'TS & GP-TS', 'Step 3')
    plt.tight_layout()
    plt.show()
