import matplotlib.pyplot as plt
import numpy as np
from tqdm.contrib.concurrent import process_map

from environments.NSClairvoyantAlgorithm import NSClairvoyantAlgorithm
from environments.NSEnvironment import NSEnvironment
from learners.SWUCB1Learner import SWUCB1Learner
from learners.CDUCBLearner import CDUCBLearner
from learners.UCB1Learner import UCB1Learner
from utils import plot_statistics

# Simulation parameters
T = 365  # n.b.: T must be equal to horizon parameter in the JSON file
n_experiments = 1000
window_size = 100

# CUSUM parameters
M = 20
eps = 0.05
h = 20
alpha = 0.01


def run_experiment(_):
    # For every experiment, we define new environment and learners
    env = NSEnvironment.from_json('data/NS_environment.json')
    # Clairvoyant
    clairvoyant = NSClairvoyantAlgorithm(env)
    # Learners
    ucb1_learner = UCB1Learner(len(env.prices))
    swucb_learner = SWUCB1Learner(len(env.prices), windows_size=window_size)
    cducb_learner = CDUCBLearner(len(env.prices), M, eps, h, alpha)

    # Data structures
    instantaneous_reward_clairvoyant = np.zeros(T)
    instantaneous_reward_ucb1 = np.zeros(T)
    instantaneous_reward_swucb = np.zeros(T)
    instantaneous_reward_cducb = np.zeros(T)

    instantaneous_regret_clairvoyant = np.zeros(T)
    instantaneous_regret_ucb1 = np.zeros(T)
    instantaneous_regret_swucb = np.zeros(T)
    instantaneous_regret_cducb = np.zeros(T)

    for t in range(T):
        # Clairvoyant algorithm
        current_phase = int(t / clairvoyant.environment.phases_size)
        opt_reward = clairvoyant.optimal_rewards[current_phase]
        opt_bid_id = clairvoyant.optimal_bids_id[current_phase]

        instantaneous_reward_clairvoyant[t] = opt_reward
        instantaneous_regret_clairvoyant[t] = 0

        # UCB1 learner
        pulled_arm = ucb1_learner.pull_arm()
        bernoulli_reward = env.round(pulled_arm, ucb1_learner.t)
        ucb1_learner.update(pulled_arm, bernoulli_reward * env.prices[pulled_arm])

        total_reward = env.compute_reward(pulled_arm, opt_bid_id, user_class=0, phase=current_phase)
        instantaneous_reward_ucb1[t] = total_reward
        regret = opt_reward - total_reward
        instantaneous_regret_ucb1[t] = regret

        # SW-UCB learner
        pulled_arm = swucb_learner.pull_arm()
        bernoulli_reward = env.round(pulled_arm, swucb_learner.t)
        swucb_learner.update(pulled_arm, bernoulli_reward * env.prices[pulled_arm])

        total_reward = env.compute_reward(pulled_arm, opt_bid_id, user_class=0, phase=current_phase)
        instantaneous_reward_swucb[t] = total_reward
        regret = opt_reward - total_reward
        instantaneous_regret_swucb[t] = regret

        # CD-UCB learner
        pulled_arm = cducb_learner.pull_arm()
        bernoulli_reward = env.round(pulled_arm, cducb_learner.t)
        cducb_learner.update(pulled_arm, bernoulli_reward * env.prices[pulled_arm])

        total_reward = env.compute_reward(pulled_arm, opt_bid_id, user_class=0, phase=current_phase)
        instantaneous_reward_cducb[t] = total_reward
        regret = opt_reward - total_reward
        instantaneous_regret_cducb[t] = regret

    return instantaneous_reward_clairvoyant, instantaneous_reward_ucb1, instantaneous_reward_swucb, \
        instantaneous_reward_cducb, instantaneous_regret_clairvoyant, instantaneous_regret_ucb1, \
        instantaneous_regret_swucb, instantaneous_regret_cducb


if __name__ == '__main__':
    # Run the experiments in parallel
    results_list = process_map(run_experiment, range(n_experiments), max_workers=10, chunksize=1)
    # Array of shape (n_experiments, n_learners * 2, T)
    results_array = np.array(results_list)

    # Extract the results into multiple arrays of shape (n_experiments, T)
    inst_reward_clairvoyant = results_array[:, 0, :]
    inst_reward_ucb1 = results_array[:, 1, :]
    inst_reward_swucb = results_array[:, 2, :]
    inst_reward_cducb = results_array[:, 3, :]
    inst_regret_clairvoyant = results_array[:, 4, :]
    inst_regret_ucb1 = results_array[:, 5, :]
    inst_regret_swucb = results_array[:, 6, :]
    inst_regret_cducb = results_array[:, 7, :]

    # Generate plots of the mean and standard deviation of the results
    plot_statistics(inst_reward_clairvoyant, inst_regret_clairvoyant, 'Clairvoyant', 'Step 5')
    plot_statistics(inst_reward_ucb1, inst_regret_ucb1, 'UCB1', 'Step 5')
    plot_statistics(inst_reward_swucb, inst_regret_swucb, 'SW-UCB', 'Step 5')
    plot_statistics(inst_reward_cducb, inst_regret_cducb, 'CD-UCB', 'Step 5')
    plt.tight_layout()
    plt.show()
