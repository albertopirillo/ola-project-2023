import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from ClairvoyantAlgorithm import ClairvoyantAlgorithm
from Environment import Environment
from TSLearner import TSLearner
from UCB1Learner import UCB1Learner
from utils import plot_statistics

# Environment parameters
num_classes = 3
num_arms = 5
bids = np.linspace(0, 100, 100)
prices = np.array([15.5, 30.7, 60.2, 70.6, 90.8])
noise_mean = 0.0
noise_std = 5.0
arms_mean = np.array([[0.4, 0.7, 0.3, 0.2, 0.1],
                      [0.1, 0.2, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.3, 0.1, 0.0]])
class_probabilities = np.array([1, 0, 0])

# Simulation parameters
T = 365
n_experiments = 100

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
        clairvoyant = ClairvoyantAlgorithm(env)
        ucb1_learner = UCB1Learner(num_arms)
        ts_learner = TSLearner(num_arms)

        for t in range(T):
            # Clairvoyant Algorithm
            best_arm_id = np.argmax(env.arms_mean, axis=1)[0]
            best_arm_mean = np.max(env.arms_mean, axis=1)[0]
            opt_reward = best_arm_mean

            instantaneous_reward_clairvoyant[e][t] = opt_reward
            instantaneous_regret_clairvoyant[e][t] = 0

            # UCB1 Learner
            pulled_arm = ucb1_learner.pull_arm()
            reward = env.round(pulled_arm)
            ucb1_learner.update(pulled_arm, reward)

            instantaneous_reward_ucb1[e][t] = reward
            regret = opt_reward - reward
            instantaneous_regret_ucb1[e][t] = regret

            # Thompson Sampling Learner
            pulled_arm = ts_learner.pull_arm()
            reward = env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward)

            instantaneous_reward_ts[e][t] = reward
            regret = opt_reward - reward
            instantaneous_regret_ts[e][t] = regret

    plot_statistics(instantaneous_reward_clairvoyant, instantaneous_regret_clairvoyant, 'Clairvoyant')
    plot_statistics(instantaneous_reward_ucb1, instantaneous_regret_ucb1, 'UCB1')
    plot_statistics(instantaneous_reward_ts, instantaneous_regret_ts, 'TS')
    plt.show()
