import matplotlib.pyplot as plt
import numpy as np
from tqdm.contrib.concurrent import process_map

from environments.ClairvoyantAlgorithm import ClairvoyantAlgorithm
from environments.Environment import Environment
from learners.TSLearner import TSLearner
from learners.UCB1Learner import UCB1Learner
from utils import plot_statistics

# Simulation parameters
T = 365
n_experiments = 5000


def run_experiment(_):
    # For every experiment, we define new environment and learners
    env = Environment.from_json('data/environment.json')
    # Advertising curves (optimal bid) are known
    clairvoyant = ClairvoyantAlgorithm(env)
    opt_reward = clairvoyant.optimal_rewards[0]
    opt_bid_id = clairvoyant.optimal_bids_id[0]
    # Learners
    ucb1_learner = UCB1Learner(len(env.prices))
    ts_learner = TSLearner(len(env.prices))

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

        # UCB1 learner
        pulled_arm = ucb1_learner.pull_arm()
        bernoulli_reward = env.round(pulled_arm)
        ucb1_learner.update(pulled_arm, bernoulli_reward * env.prices[pulled_arm])

        total_reward = env.compute_reward(pulled_arm, opt_bid_id, user_class=0)
        instantaneous_reward_ucb1[t] = total_reward
        regret = opt_reward - total_reward
        instantaneous_regret_ucb1[t] = regret

        # Thompson Sampling learner
        pulled_arm = ts_learner.pull_arm(env.prices)
        bernoulli_reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, bernoulli_reward)

        total_reward = env.compute_reward(pulled_arm, opt_bid_id, user_class=0)
        instantaneous_reward_ts[t] = total_reward
        regret = opt_reward - total_reward
        instantaneous_regret_ts[t] = regret

    return instantaneous_reward_clairvoyant, instantaneous_reward_ucb1, instantaneous_reward_ts, \
        instantaneous_regret_clairvoyant, instantaneous_regret_ucb1, instantaneous_regret_ts


if __name__ == '__main__':
    # Run the experiments in parallel
    results_list = process_map(run_experiment, range(n_experiments), max_workers=10, chunksize=1)
    # Array of shape (n_experiments, n_learners * 2, T)
    results_array = np.array(results_list)

    # Extract the results into multiple arrays of shape (n_experiments, T)
    inst_reward_clairvoyant = results_array[:, 0, :]
    inst_reward_ucb1 = results_array[:, 1, :]
    inst_reward_ts = results_array[:, 2, :]
    inst_regret_clairvoyant = results_array[:, 3, :]
    inst_regret_ucb1 = results_array[:, 4, :]
    inst_regret_ts = results_array[:, 5, :]

    # Generate plots of the mean and standard deviation of the results
    plot_statistics(inst_reward_clairvoyant, inst_regret_clairvoyant, 'Clairvoyant', 'Step 1')
    plot_statistics(inst_reward_ucb1, inst_regret_ucb1, 'UCB1', 'Step 1')
    plot_statistics(inst_reward_ts, inst_regret_ts, 'TS', 'Step 1')
    plt.tight_layout()
    plt.show()
