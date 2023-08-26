import matplotlib.pyplot as plt
import numpy as np
from tqdm.contrib.concurrent import process_map

from environments.ClairvoyantAlgorithm import ClairvoyantAlgorithm
from environments.Environment import Environment
from learners.GPTSLearner import GPTSLearner
from learners.GPUCBLearner import GPUCBLearner
from learners.TSLearner import TSLearner
from learners.UCB1Learner import UCB1Learner
from utils import plot_statistics

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Simulation parameters
T = 365
n_experiments = 10


def run_experiment(_):
    # For every experiment, we define new environment and learners
    env = Environment.from_json('data/environment_step4.json')
    # No information on pricing nor advertising
    clairvoyant = ClairvoyantAlgorithm(env)
    # Learners
    ucb1_learner1 = UCB1Learner(len(env.prices))
    ts_learner1 = TSLearner(len(env.prices))
    gp_ucb_learner1 = GPUCBLearner(len(env.bids), env.bids)
    gp_ts_learner1 = GPTSLearner(len(env.bids), env.bids)

    ucb1_learner2 = UCB1Learner(len(env.prices))
    ts_learner2 = TSLearner(len(env.prices))
    gp_ucb_learner2 = GPUCBLearner(len(env.bids), env.bids)
    gp_ts_learner2 = GPTSLearner(len(env.bids), env.bids)

    ucb1_learner3 = UCB1Learner(len(env.prices))
    ts_learner3 = TSLearner(len(env.prices))
    gp_ucb_learner3 = GPUCBLearner(len(env.bids), env.bids)
    gp_ts_learner3 = GPTSLearner(len(env.bids), env.bids)

    # Data structures
    instantaneous_reward_clairvoyant = np.zeros(T)
    instantaneous_reward_ucb1 = np.zeros(T)
    instantaneous_reward_ts = np.zeros(T)

    instantaneous_regret_clairvoyant = np.zeros(T)
    instantaneous_regret_ucb1 = np.zeros(T)
    instantaneous_regret_ts = np.zeros(T)

    for t in range(T):
        # Clairvoyant algorithm
        extracted_class: int = np.random.choice(env.num_classes, p=env.class_probabilities)
        features: np.ndarray[int] = np.zeros(2)
        match extracted_class:
            case 0:
                features[0] = 0
                features[1] = 0
            case 1:
                features[0] = np.random.choice(np.arange(2), size=1)
                features[1] = 1 - features[0]
            case 2:
                features[0] = 1
                features[1] = 1

        opt_reward = clairvoyant.optimal_rewards[extracted_class]
        instantaneous_reward_clairvoyant[t] = opt_reward
        instantaneous_regret_clairvoyant[t] = 0

        # UCB1 and GP-UCB learners
        if features[0] == 0 and features[1] == 0:
            pulled_arm_pricing = ucb1_learner1.pull_arm()
            bernoulli_reward = env.round_step4(pulled_arm_pricing, extracted_class)
            ucb1_learner1.update(pulled_arm_pricing, bernoulli_reward * env.prices[pulled_arm_pricing])
            pulled_arm_advertising = gp_ucb_learner1.pull_arm()
            total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising, user_class=0)
            gp_ucb_learner1.update(pulled_arm_advertising, total_reward)

        elif features[0] == 1 and features[1] == 1:
            pulled_arm_pricing = ucb1_learner3.pull_arm()
            bernoulli_reward = env.round_step4(pulled_arm_pricing, extracted_class)
            ucb1_learner3.update(pulled_arm_pricing, bernoulli_reward * env.prices[pulled_arm_pricing])
            pulled_arm_advertising = gp_ucb_learner3.pull_arm()
            total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising, user_class=2)
            gp_ucb_learner3.update(pulled_arm_advertising, total_reward)

        else:
            pulled_arm_pricing = ucb1_learner2.pull_arm()
            bernoulli_reward = env.round_step4(pulled_arm_pricing, extracted_class)
            ucb1_learner2.update(pulled_arm_pricing, bernoulli_reward * env.prices[pulled_arm_pricing])
            pulled_arm_advertising = gp_ucb_learner2.pull_arm()
            total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising, user_class=1)
            gp_ucb_learner2.update(pulled_arm_advertising, total_reward)

        instantaneous_reward_ucb1[t] = total_reward
        regret = opt_reward - total_reward
        instantaneous_regret_ucb1[t] = regret

        # TS and GP-TS learners
        if features[0] == 0 and features[1] == 0:
            pulled_arm_pricing = ts_learner1.pull_arm(env.prices)
            bernoulli_reward = env.round_step4(pulled_arm_pricing, extracted_class)
            ts_learner1.update(pulled_arm_pricing, bernoulli_reward)
            pulled_arm_advertising = gp_ts_learner1.pull_arm()
            total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising, user_class=0)
            gp_ts_learner1.update(pulled_arm_advertising, total_reward)

        elif features[0] == 1 and features[1] == 1:
            pulled_arm_pricing = ts_learner3.pull_arm(env.prices)
            bernoulli_reward = env.round_step4(pulled_arm_pricing, extracted_class)
            ts_learner3.update(pulled_arm_pricing, bernoulli_reward)
            pulled_arm_advertising = gp_ts_learner3.pull_arm()
            total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising, user_class=2)
            gp_ts_learner3.update(pulled_arm_advertising, total_reward)

        else:
            pulled_arm_pricing = ts_learner2.pull_arm(env.prices)
            bernoulli_reward = env.round_step4(pulled_arm_pricing, extracted_class)
            ts_learner2.update(pulled_arm_pricing, bernoulli_reward)
            pulled_arm_advertising = gp_ts_learner2.pull_arm()
            total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising, user_class=1)
            gp_ts_learner2.update(pulled_arm_advertising, total_reward)

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
    plot_statistics(inst_reward_clairvoyant, inst_regret_clairvoyant, 'Clairvoyant', 'Step 4.1 - Context known')
    plot_statistics(inst_reward_ucb1, inst_regret_ucb1, 'UCB1 & GP-UCB', 'Step 4.1')
    plot_statistics(inst_reward_ts, inst_regret_ts, 'TS & GP-TS', 'Step 4.1')
    plt.tight_layout()
    plt.show()
