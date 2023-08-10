import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from NSClairvoyantAlgorithm import NSClairvoyantAlgorithm
from NSEnvironment import NSEnvironment
from SWUCB1Learner import SWUCB1Learner
from CDUCBLearner import CDUCBLearner
from UCB1Learner import UCB1Learner
from utils import plot_statistics


# Simulation parameters
T = 360                 #n.b. : T must be equal to horizon parameter in 
n_experiments = 100
windows_size = int(100)

# CUSUM parameters
M = 100
eps = 0.05
h = 20
alpha = 0.01

# History
instantaneous_reward_clairvoyant = np.zeros(shape=(n_experiments, T))
instantaneous_reward_ucb1 = np.zeros(shape=(n_experiments, T))
instantaneous_reward_swucb = np.zeros(shape=(n_experiments, T))
instantaneous_reward_cducb = np.zeros(shape=(n_experiments, T))

instantaneous_regret_clairvoyant = np.zeros(shape=(n_experiments, T))
instantaneous_regret_ucb1 = np.zeros(shape=(n_experiments, T))
instantaneous_regret_swucb = np.zeros(shape=(n_experiments, T))
instantaneous_regret_cducb = np.zeros(shape=(n_experiments, T))

if __name__ == '__main__':
    for e in trange(n_experiments):
        # For every experiment, we define new environment and learners
        env = NSEnvironment.from_json('data/NSenvironment.json')
        # Clairvoyant
        clairvoyant = NSClairvoyantAlgorithm(env)
        # Learners
        ucb1_learner = UCB1Learner(len(env.prices))
        swucb_learner = SWUCB1Learner(len(env.prices), windows_size=windows_size)
        cducb_learner = CDUCBLearner(len(env.prices), M, eps, h, alpha)

        for t in range(T):
            # Clairvoyant algorithm
            current_phase = int(t / clairvoyant.environment.phases_size)
            opt_reward = clairvoyant.optimal_rewards[current_phase]
            opt_bid_id = clairvoyant.optimal_bids_id[current_phase]

            instantaneous_reward_clairvoyant[e][t] = opt_reward
            instantaneous_regret_clairvoyant[e][t] = 0


            # UCB1 learner
            pulled_arm = ucb1_learner.pull_arm()
            pricing_reward = env.round(pulled_arm, ucb1_learner.t)
            ucb1_learner.update(pulled_arm, pricing_reward)

            total_reward = env.compute_reward(pulled_arm, opt_bid_id, user_class=0, phase=current_phase)
            instantaneous_reward_ucb1[e][t] = total_reward
            regret = opt_reward - total_reward
            instantaneous_regret_ucb1[e][t] = regret
            
            # SWUCB learner
            pulled_arm = swucb_learner.pull_arm()
            pricing_reward = env.round(pulled_arm, swucb_learner.t)
            swucb_learner.update(pulled_arm, pricing_reward)

            total_reward = env.compute_reward(pulled_arm, opt_bid_id, user_class=0, phase=current_phase)
            instantaneous_reward_swucb[e][t] = total_reward
            regret = opt_reward - total_reward
            instantaneous_regret_swucb[e][t] = regret

            # CDUCB learner
            pulled_arm = cducb_learner.pull_arm()
            pricing_reward = env.round(pulled_arm, cducb_learner.t)
            cducb_learner.update(pulled_arm, pricing_reward)

            total_reward = env.compute_reward(pulled_arm, opt_bid_id, user_class=0, phase=current_phase)
            instantaneous_reward_cducb[e][t] = total_reward
            regret = opt_reward - total_reward
            instantaneous_regret_cducb[e][t] = regret



    plot_statistics(instantaneous_reward_clairvoyant, instantaneous_regret_clairvoyant, 'Clairvoyant', 'Step 5')
    plot_statistics(instantaneous_reward_ucb1, instantaneous_regret_ucb1, 'UCB1', 'Step 5')
    plot_statistics(instantaneous_reward_swucb, instantaneous_regret_swucb, 'SW-UCB1', 'Step 5')
    plot_statistics(instantaneous_reward_cducb, instantaneous_regret_cducb, 'CD-UCB1', 'Step 5')
    plt.tight_layout()
    plt.show()