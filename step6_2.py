import matplotlib.pyplot as plt
import numpy as np
from tqdm.contrib.concurrent import process_map

from environments.NSClairvoyantAlgorithm import NSClairvoyantAlgorithm
from environments.NSEnvironment import NSEnvironment
from learners.SWUCB1Learner import SWUCB1Learner
from learners.CDUCBLearner import CDUCBLearner
from learners.UCB1Learner import UCB1Learner
from learners.EXP3Learner import EXP3Learner
from utils import plot_statistics

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

# Simulation parameters
T = 365  # n.b.: T must be equal to horizon parameter in the JSON file
n_experiments = 5000
window_size = 100

# CUSUM parameters
M = 50
eps = 0.5
h = 40
alpha = 0.1

# EXP3 parameter
gamma = 0.1


def run_experiment(_):
    # For every experiment, we define new environment and learners
    env = NSEnvironment.from_json('data/NS_environment_step6_2.json')

    # Clairvoyant
    clairvoyant = NSClairvoyantAlgorithm(env)
    # Learners
    ucb1_learner = UCB1Learner(len(env.prices))
    sw_ucb_learner = SWUCB1Learner(len(env.prices), windows_size=window_size)
    cd_ucb_learner = CDUCBLearner(len(env.prices), M, eps, h, alpha)
    exp3_learner = EXP3Learner(len(env.prices), gamma)

    # Data structures
    instantaneous_reward_clairvoyant = np.zeros(T)
    instantaneous_reward_exp3 = np.zeros(T)
    instantaneous_reward_ucb1 = np.zeros(T)
    instantaneous_reward_sw_ucb = np.zeros(T)
    instantaneous_reward_cd_ucb = np.zeros(T)

    instantaneous_regret_clairvoyant = np.zeros(T)
    instantaneous_regret_exp3 = np.zeros(T)
    instantaneous_regret_ucb1 = np.zeros(T)
    instantaneous_regret_sw_ucb = np.zeros(T)
    instantaneous_regret_cd_ucb = np.zeros(T)

    for t in range(T):
        # Clairvoyant algorithm
        current_phase = (t // env.phase_length) % env.n_phases
        opt_reward = clairvoyant.optimal_rewards[current_phase]
        opt_bid_id = clairvoyant.optimal_bids_id[current_phase]

        instantaneous_reward_clairvoyant[t] = opt_reward
        instantaneous_regret_clairvoyant[t] = 0

        # EXP3 learner
        pulled_arm = exp3_learner.pull_arm()
        bernoulli_reward = env.round(pulled_arm, exp3_learner.t)
        exp3_learner.update(pulled_arm, float(bernoulli_reward * env.prices[pulled_arm]))

        total_reward = env.compute_reward(pulled_arm, opt_bid_id, user_class=0, phase=current_phase)
        instantaneous_reward_exp3[t] = total_reward
        regret = opt_reward - total_reward
        instantaneous_regret_exp3[t] = regret

        # UCB1 learner
        pulled_arm = ucb1_learner.pull_arm()
        bernoulli_reward = env.round(pulled_arm, ucb1_learner.t)
        ucb1_learner.update(pulled_arm, float(bernoulli_reward * env.prices[pulled_arm]))

        total_reward = env.compute_reward(pulled_arm, opt_bid_id, user_class=0, phase=current_phase)
        instantaneous_reward_ucb1[t] = total_reward
        regret = opt_reward - total_reward
        instantaneous_regret_ucb1[t] = regret

        # SW-UCB learner
        pulled_arm = sw_ucb_learner.pull_arm()
        bernoulli_reward = env.round(pulled_arm, sw_ucb_learner.t)
        sw_ucb_learner.update(pulled_arm, float(bernoulli_reward * env.prices[pulled_arm]))

        total_reward = env.compute_reward(pulled_arm, opt_bid_id, user_class=0, phase=current_phase)
        instantaneous_reward_sw_ucb[t] = total_reward
        regret = opt_reward - total_reward
        instantaneous_regret_sw_ucb[t] = regret

        # CD-UCB learner
        pulled_arm = cd_ucb_learner.pull_arm()
        bernoulli_reward = env.round(pulled_arm, cd_ucb_learner.t)
        cd_ucb_learner.update(pulled_arm, float(bernoulli_reward * env.prices[pulled_arm]))

        total_reward = env.compute_reward(pulled_arm, opt_bid_id, user_class=0, phase=current_phase)
        instantaneous_reward_cd_ucb[t] = total_reward
        regret = opt_reward - total_reward
        instantaneous_regret_cd_ucb[t] = regret

    return instantaneous_reward_clairvoyant,  instantaneous_reward_exp3, instantaneous_reward_ucb1, \
        instantaneous_reward_sw_ucb, instantaneous_reward_cd_ucb, instantaneous_regret_clairvoyant, \
        instantaneous_regret_exp3, instantaneous_regret_ucb1, instantaneous_regret_sw_ucb, instantaneous_regret_cd_ucb


if __name__ == '__main__':
    # Run the experiments in parallel
    results_list = process_map(run_experiment, range(n_experiments), max_workers=10, chunksize=1)
    # Array of shape (n_experiments, n_learners * 2, T)
    results_array = np.array(results_list)

    # Extract the results into multiple arrays of shape (n_experiments, T)
    inst_reward_clairvoyant = results_array[:, 0, :]
    inst_reward_exp3 = results_array[:, 1, :]
    inst_reward_ucb1 = results_array[:, 2, :]
    inst_reward_sw_ucb = results_array[:, 3, :]
    inst_reward_cd_ucb = results_array[:, 4, :]
    inst_regret_clairvoyant = results_array[:, 5, :]
    inst_regret_exp3 = results_array[:, 6, :]
    inst_regret_ucb1 = results_array[:, 7, :]
    inst_regret_sw_ucb = results_array[:, 8, :]
    inst_regret_cd_ucb = results_array[:, 9, :]

    # Generate plots of the mean and standard deviation of the results
    plot_statistics(inst_reward_clairvoyant, inst_regret_clairvoyant, 'Clairvoyant', 'Step 6.2')
    plot_statistics(inst_reward_exp3, inst_regret_exp3, 'EXP3', 'Step 6.2')
    plot_statistics(inst_reward_ucb1, inst_regret_ucb1, 'UCB1', 'Step 6.2')
    plot_statistics(inst_reward_sw_ucb, inst_regret_sw_ucb, 'SW-UCB', 'Step 6.2')
    plot_statistics(inst_reward_cd_ucb, inst_regret_cd_ucb, 'CD-UCB', 'Step 6.2')
    plt.tight_layout()
    plt.show()
