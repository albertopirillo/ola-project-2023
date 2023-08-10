import matplotlib.pyplot as plt
import numpy as np
from tqdm.contrib.concurrent import process_map

from ClairvoyantAlgorithm import ClairvoyantAlgorithm
from Environment import Environment
from GPTSLearner import GPTSLearner
from GPUCBLearner import GPUCBLearner
from TSLearner import TSLearner
from UCB1Learner import UCB1Learner
from utils import plot_statistics

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Simulation parameters
T = 200
n_experiments = 100


def run_experiment(_):
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

    # Data structures
    instantaneous_reward_clairvoyant = np.zeros(T)
    instantaneous_reward_ucb1 = np.zeros(T)
    instantaneous_reward_ts = np.zeros(T)

    instantaneous_regret_clairvoyant = np.zeros(T)
    instantaneous_regret_ucb1 = np.zeros(T)
    instantaneous_regret_ts = np.zeros(T)

    for t in range(T):
        # Clairvoyant algorithm
        instantaneous_reward_clairvoyant[t] = opt_reward
        instantaneous_regret_clairvoyant[t] = 0

        # UCB1 and GP-UCB learners
        pulled_arm_pricing = ucb1_learner.pull_arm()
        pricing_reward = env.round(pulled_arm_pricing)
        ucb1_learner.update(pulled_arm_pricing, pricing_reward)

        pulled_arm_advertising = gp_ucb_learner.pull_arm()
        total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising, user_class=0)
        gp_ucb_learner.update(pulled_arm_advertising, total_reward)

        instantaneous_reward_ucb1[t] = total_reward
        regret = opt_reward - total_reward
        instantaneous_regret_ucb1[t] = regret

        # TS and GP-TS learners
        pulled_arm_pricing = ts_learner.pull_arm()
        pricing_reward = env.round(pulled_arm_pricing)
        ts_learner.update(pulled_arm_pricing, pricing_reward)

        pulled_arm_advertising = gp_ts_learner.pull_arm()
        total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising, user_class=0)
        gp_ts_learner.update(pulled_arm_advertising, total_reward)

        instantaneous_reward_ts[t] = total_reward
        regret = opt_reward - total_reward
        instantaneous_regret_ts[t] = regret

    return instantaneous_reward_clairvoyant, instantaneous_reward_ucb1, instantaneous_reward_ts, \
        instantaneous_regret_clairvoyant, instantaneous_regret_ucb1, instantaneous_regret_ts


if __name__ == '__main__':
    # Run the experiments in parallel
    results_list = process_map(run_experiment, range(n_experiments), max_workers=10, chunksize=1)
    # Array of shape (n_experiments, 6, T)
    results_array = np.array(results_list)

    # Extract the results into six arrays of shape (n_experiments, T)
    inst_reward_clairvoyant = results_array[:, 0, :]
    inst_reward_ucb1 = results_array[:, 1, :]
    inst_reward_ts = results_array[:, 2, :]
    inst_regret_clairvoyant = results_array[:, 3, :]
    inst_regret_ucb1 = results_array[:, 4, :]
    inst_regret_ts = results_array[:, 5, :]

    # Generate plots of the mean and standard deviation of the results
    plot_statistics(inst_reward_clairvoyant, inst_regret_clairvoyant, 'Clairvoyant', 'Step 3')
    plot_statistics(inst_reward_ucb1, inst_regret_ucb1, 'UCB1 & GP-UCB', 'Step 3')
    plot_statistics(inst_reward_ts, inst_regret_ts, 'TS & GP-TS', 'Step 3')
    plt.tight_layout()
    plt.show()
