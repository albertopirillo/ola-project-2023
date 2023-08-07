import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

# import warnings
# from sklearn.exceptions import ConvergenceWarning
# warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

from ClairvoyantAlgorithm import ClairvoyantAlgorithm
from Environment import Environment
from GPTSLearner import GPTSLearner
from GPUCBLearner import GPUCBLearner
from utils import plot_statistics

# Environment parameters
num_classes = 3
num_arms = 100
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
    opt_arm_id = np.argmax(arms_mean, axis=1)[0]
    opt_conv_rate = np.max(arms_mean, axis=1)[0]
    opt_price = prices[opt_arm_id]

    for e in trange(n_experiments):
        # For every experiment, we define new environment and learners
        env = Environment(num_classes, bids, prices, noise_mean, noise_std, arms_mean, class_probabilities)
        observations = env.generate_observation_from_click(bids, user_class=0)
        clairvoyant = ClairvoyantAlgorithm(env)
        gp_ucb_learner = GPUCBLearner(num_arms, observations)
        gp_ts_learner = GPTSLearner(num_arms, observations)

        for t in trange(T):
            # Clairvoyant Algorithm
            opt_reward = clairvoyant.compute_optimal_solution()

            instantaneous_reward_clairvoyant[e][t] = opt_reward
            instantaneous_regret_clairvoyant[e][t] = 0

            # GP-UCB Learner
            pulled_arm = gp_ucb_learner.pull_arm()
            reward = compute_reward(opt_conv_rate, opt_price, bids[pulled_arm])
            gp_ucb_learner.update(pulled_arm, reward)

            instantaneous_reward_ucb1[e][t] = reward
            regret = opt_reward - reward
            instantaneous_regret_ucb1[e][t] = regret

            # GP Thompson Sampling Learner
            pulled_arm = gp_ts_learner.pull_arm()
            reward = compute_reward(opt_conv_rate, opt_price, bids[pulled_arm])
            gp_ts_learner.update(pulled_arm, reward)

            instantaneous_reward_ts[e][t] = reward
            regret = opt_reward - reward
            instantaneous_regret_ts[e][t] = regret

    plot_statistics(instantaneous_reward_clairvoyant, instantaneous_regret_clairvoyant, 'Clairvoyant', 'Step 2')
    plot_statistics(instantaneous_reward_ucb1, instantaneous_regret_ucb1, 'GP-UCB', 'Step 2')
    plot_statistics(instantaneous_reward_ts, instantaneous_regret_ts, 'GP-TS', 'Step 2')
    plt.show()