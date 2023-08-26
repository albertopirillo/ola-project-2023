from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
from tqdm.contrib.concurrent import process_map

from environments.ClairvoyantAlgorithm import ClairvoyantAlgorithm
from environments.Environment import Environment
from learners.GPTSLearner import GPTSLearner
from learners.GPUCBLearner import GPUCBLearner
from learners.TSLearner import TSLearner
from learners.UCB1Learner import UCB1Learner
from utils import plot_statistics, hoeffding_bound

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Simulation parameters
T = 365
n_experiments = 10


class ChoiceOfContext(Enum):
    NO_CHOICE = auto(),
    ONE_CLASS = auto(),
    TWO_CLASSES_F1 = auto(),
    TWO_CLASSES_F2 = auto(),
    FOUR_CLASSES = auto(),


def run_experiment(_):
    # For every experiment, we define new environment and learners
    env = Environment.from_json('data/environment_step4.json')
    # No information on pricing nor advertising
    clairvoyant = ClairvoyantAlgorithm(env)

    # Learners

    # Context With a Single Class Week 1 & 2
    ucb1_learner = UCB1Learner(len(env.prices))
    ts_learner = TSLearner(len(env.prices))
    gp_ucb_learner = GPUCBLearner(len(env.bids), env.bids)
    gp_ts_learner = GPTSLearner(len(env.bids), env.bids)

    # Context With Two Classes Split on Feature 1 Week 3 & 4
    ucb1_learner_feature1_1 = UCB1Learner(len(env.prices))
    ts_learner_feature1_1 = TSLearner(len(env.prices))
    gp_ucb_learner_feature1_1 = GPUCBLearner(len(env.bids), env.bids)
    gp_ts_learner_feature1_1 = GPTSLearner(len(env.bids), env.bids)

    ucb1_learner_feature1_0 = UCB1Learner(len(env.prices))
    ts_learner_feature1_0 = TSLearner(len(env.prices))
    gp_ucb_learner_feature1_0 = GPUCBLearner(len(env.bids), env.bids)
    gp_ts_learner_feature1_0 = GPTSLearner(len(env.bids), env.bids)

    # Context With Two Classes Split on Feature 2 Week 5 & 6
    ucb1_learner_feature2_1 = UCB1Learner(len(env.prices))
    ts_learner_feature2_1 = TSLearner(len(env.prices))
    gp_ucb_learner_feature2_1 = GPUCBLearner(len(env.bids), env.bids)
    gp_ts_learner_feature2_1 = GPTSLearner(len(env.bids), env.bids)

    ucb1_learner_feature2_0 = UCB1Learner(len(env.prices))
    ts_learner_feature2_0 = TSLearner(len(env.prices))
    gp_ucb_learner_feature2_0 = GPUCBLearner(len(env.bids), env.bids)
    gp_ts_learner_feature2_0 = GPTSLearner(len(env.bids), env.bids)

    # Context With Four Classes Week 5 & 6
    ucb1_learner_c1 = UCB1Learner(len(env.prices))
    ts_learner_c1 = TSLearner(len(env.prices))
    gp_ucb_learner_c1 = GPUCBLearner(len(env.bids), env.bids)
    gp_ts_learner_c1 = GPTSLearner(len(env.bids), env.bids)

    ucb1_learner_c2 = UCB1Learner(len(env.prices))
    ts_learner_c2 = TSLearner(len(env.prices))
    gp_ucb_learner_c2 = GPUCBLearner(len(env.bids), env.bids)
    gp_ts_learner_c2 = GPTSLearner(len(env.bids), env.bids)

    ucb1_learner_c3 = UCB1Learner(len(env.prices))
    ts_learner_c3 = TSLearner(len(env.prices))
    gp_ucb_learner_c3 = GPUCBLearner(len(env.bids), env.bids)
    gp_ts_learner_c3 = GPTSLearner(len(env.bids), env.bids)

    ucb1_learner_c4 = UCB1Learner(len(env.prices))
    ts_learner_c4 = TSLearner(len(env.prices))
    gp_ucb_learner_c4 = GPUCBLearner(len(env.bids), env.bids)
    gp_ts_learner_c4 = GPTSLearner(len(env.bids), env.bids)

    # Data structures
    instantaneous_reward_clairvoyant = np.zeros(T)
    instantaneous_reward_ucb1 = np.zeros(T)
    instantaneous_reward_ts = np.zeros(T)

    instantaneous_regret_clairvoyant = np.zeros(T)
    instantaneous_regret_ucb1 = np.zeros(T)
    instantaneous_regret_ts = np.zeros(T)

    choice_of_context = ChoiceOfContext.NO_CHOICE
    week = 7

    confidence = 0.8
    z = 14  # number of samples in a period of context

    for t in range(T):

        # Decision after 6 Weeks between contexts with 1 or 2 Classes (After compute the Best Feature)
        if t == week * 6:
            best_value_1class = ucb1_learner.get_best_expected_value()

            best_value_feature1_0 = ucb1_learner_feature1_0.get_best_expected_value()
            best_value_feature1_1 = ucb1_learner_feature1_1.get_best_expected_value()
            best_value_feature2_0 = ucb1_learner_feature2_0.get_best_expected_value()
            best_value_feature2_1 = ucb1_learner_feature2_1.get_best_expected_value()

            hoef_feature1_0 = hoeffding_bound(best_value_feature1_0, confidence, z)
            hoef_feature1_1 = hoeffding_bound(best_value_feature1_1, confidence, z)
            hoef_feature2_0 = hoeffding_bound(best_value_feature2_0, confidence, z)
            hoef_feature2_1 = hoeffding_bound(best_value_feature2_1, confidence, z)

            best_value_2class_feature1 = hoef_feature1_0 * best_value_feature1_0 + hoef_feature1_1 * best_value_feature1_1
            best_value_2class_feature2 = hoef_feature2_0 * best_value_feature2_0 + hoef_feature2_1 * best_value_feature2_1

            hoef_best_value_2class = np.maximum(best_value_2class_feature1, best_value_2class_feature2)
            best_feature = int(np.argmax([best_value_2class_feature1, best_value_2class_feature2]))

            if hoef_best_value_2class <= best_value_1class:
                choice_of_context = ChoiceOfContext.ONE_CLASS
            else:
                choice_of_context = ChoiceOfContext.NO_CHOICE
                confidence = 0.6

        # Decision after 8 Weeks between contexts with 2 or 4 Classes
        if t == week * 8 and choice_of_context == ChoiceOfContext.NO_CHOICE:

            if best_feature == 0:
                best_value_2class = best_value_feature1_0 + best_value_feature1_1
            else:
                best_value_2class = best_value_feature2_0 + best_value_feature2_1

            best_value_c1 = ucb1_learner_c1.get_best_expected_value()
            best_value_c2 = ucb1_learner_c2.get_best_expected_value()
            best_value_c3 = ucb1_learner_c3.get_best_expected_value()
            best_value_c4 = ucb1_learner_c4.get_best_expected_value()

            hoef_c1 = hoeffding_bound(best_value_c1, confidence, z)
            hoef_c2 = hoeffding_bound(best_value_c2, confidence, z)
            hoef_c3 = hoeffding_bound(best_value_c3, confidence, z)
            hoef_c4 = hoeffding_bound(best_value_c4, confidence, z)

            hoef_best_value_4class = hoef_c1 * best_value_c1 + hoef_c2 * best_value_c2 + hoef_c3 * best_value_c3 + hoef_c4 * best_value_c4

            if hoef_best_value_4class <= best_value_2class and best_feature == 0:
                choice_of_context = ChoiceOfContext.TWO_CLASSES_F1

            elif hoef_best_value_4class <= best_value_2class:
                choice_of_context = ChoiceOfContext.TWO_CLASSES_F2

            else:
                choice_of_context = ChoiceOfContext.FOUR_CLASSES

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

        # Clairvoyant algorithm
        opt_reward = clairvoyant.optimal_rewards[extracted_class]
        instantaneous_reward_clairvoyant[t] = opt_reward
        instantaneous_regret_clairvoyant[t] = 0

        # Firsts 2 Weeks Estimation of One Class or Decision of ending the experiment with 1 Class
        if t < week * 2 or choice_of_context == ChoiceOfContext.ONE_CLASS:

            # UCB1 and GP-UCB learners
            pulled_arm_pricing = ucb1_learner.pull_arm()
            bernoulli_reward = env.round_step4(pulled_arm_pricing, extracted_class)
            ucb1_learner.update(pulled_arm_pricing, bernoulli_reward * env.prices[pulled_arm_pricing])

            pulled_arm_advertising = gp_ucb_learner.pull_arm()
            total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising, user_class=extracted_class)
            gp_ucb_learner.update(pulled_arm_advertising, total_reward)

            instantaneous_reward_ucb1[t] = total_reward
            regret = opt_reward - total_reward
            instantaneous_regret_ucb1[t] = regret

            # TS and GP-TS learners
            pulled_arm_pricing = ts_learner.pull_arm(env.prices)
            pricing_reward = env.round_step4(pulled_arm_pricing, extracted_class)
            ts_learner.update(pulled_arm_pricing, pricing_reward)

            pulled_arm_advertising = gp_ts_learner.pull_arm()
            total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising, user_class=extracted_class)
            gp_ts_learner.update(pulled_arm_advertising, total_reward)

            instantaneous_reward_ts[t] = total_reward
            regret = opt_reward - total_reward
            instantaneous_regret_ts[t] = regret

        # 3 and 4 Week Estimation of 2 Classes split on feature 1 or Decision of ending the experiment with 2 Classes split on feature 1
        elif (week * 2 <= t < week * 4) or choice_of_context == ChoiceOfContext.TWO_CLASSES_F1:
            # UCB1 and GP-UCB learners
            if features[0] == 0:
                pulled_arm_pricing = ucb1_learner_feature1_0.pull_arm()
                bernoulli_reward = env.round_step4(pulled_arm_pricing, extracted_class)
                ucb1_learner_feature1_0.update(pulled_arm_pricing, bernoulli_reward * env.prices[pulled_arm_pricing])
                pulled_arm_advertising = gp_ucb_learner_feature1_0.pull_arm()
                total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising,
                                                  user_class=extracted_class)
                gp_ucb_learner_feature1_0.update(pulled_arm_advertising, total_reward)

            else:
                pulled_arm_pricing = ucb1_learner_feature1_1.pull_arm()
                bernoulli_reward = env.round_step4(pulled_arm_pricing, extracted_class)
                ucb1_learner_feature1_1.update(pulled_arm_pricing, bernoulli_reward * env.prices[pulled_arm_pricing])
                pulled_arm_advertising = gp_ucb_learner_feature1_1.pull_arm()
                total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising,
                                                  user_class=extracted_class)
                gp_ucb_learner_feature1_1.update(pulled_arm_advertising, total_reward)

            instantaneous_reward_ucb1[t] = total_reward
            regret = opt_reward - total_reward
            instantaneous_regret_ucb1[t] = regret

            # TS and GP-TS learners
            if features[0] == 0:
                pulled_arm_pricing = ts_learner_feature1_0.pull_arm(env.prices)
                pricing_reward = env.round_step4(pulled_arm_pricing, extracted_class)
                ts_learner_feature1_0.update(pulled_arm_pricing, pricing_reward)
                pulled_arm_advertising = gp_ts_learner_feature1_0.pull_arm()
                total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising,
                                                  user_class=extracted_class)
                gp_ts_learner_feature1_0.update(pulled_arm_advertising, total_reward)

            else:
                pulled_arm_pricing = ts_learner_feature1_1.pull_arm(env.prices)
                pricing_reward = env.round_step4(pulled_arm_pricing, extracted_class)
                ts_learner_feature1_1.update(pulled_arm_pricing, pricing_reward)
                pulled_arm_advertising = gp_ts_learner_feature1_1.pull_arm()
                total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising,
                                                  user_class=extracted_class)
                gp_ts_learner_feature1_1.update(pulled_arm_advertising, total_reward)

            instantaneous_reward_ts[t] = total_reward
            regret = opt_reward - total_reward
            instantaneous_regret_ts[t] = regret

        # 5 and 6 Week Estimation of 2 Classes split on feature 2 or Decision of ending the experiment with 2 Classes split on feature 2
        elif (week * 4 <= t < week * 6) or choice_of_context == ChoiceOfContext.TWO_CLASSES_F2:
            # UCB1 and GP-UCB learners
            if features[1] == 0:
                pulled_arm_pricing = ucb1_learner_feature2_0.pull_arm()
                bernoulli_reward = env.round_step4(pulled_arm_pricing, extracted_class)
                ucb1_learner_feature2_0.update(pulled_arm_pricing, bernoulli_reward * env.prices[pulled_arm_pricing])
                pulled_arm_advertising = gp_ucb_learner_feature2_0.pull_arm()
                total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising,
                                                  user_class=extracted_class)
                gp_ucb_learner_feature2_0.update(pulled_arm_advertising, total_reward)

            else:
                pulled_arm_pricing = ucb1_learner_feature2_1.pull_arm()
                bernoulli_reward = env.round_step4(pulled_arm_pricing, extracted_class)
                ucb1_learner_feature2_1.update(pulled_arm_pricing, bernoulli_reward * env.prices[pulled_arm_pricing])
                pulled_arm_advertising = gp_ucb_learner_feature2_1.pull_arm()
                total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising,
                                                  user_class=extracted_class)
                gp_ucb_learner_feature2_1.update(pulled_arm_advertising, total_reward)

            instantaneous_reward_ucb1[t] = total_reward
            regret = opt_reward - total_reward
            instantaneous_regret_ucb1[t] = regret

            # TS and GP-TS learners
            if features[1] == 0:
                pulled_arm_pricing = ts_learner_feature2_0.pull_arm(env.prices)
                pricing_reward = env.round_step4(pulled_arm_pricing, extracted_class)
                ts_learner_feature2_0.update(pulled_arm_pricing, pricing_reward)
                pulled_arm_advertising = gp_ts_learner_feature2_0.pull_arm()
                total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising,
                                                  user_class=extracted_class)
                gp_ts_learner_feature2_0.update(pulled_arm_advertising, total_reward)
            else:
                pulled_arm_pricing = ts_learner_feature2_1.pull_arm(env.prices)
                pricing_reward = env.round_step4(pulled_arm_pricing, extracted_class)
                ts_learner_feature2_1.update(pulled_arm_pricing, pricing_reward)
                pulled_arm_advertising = gp_ts_learner_feature2_1.pull_arm()
                total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising,
                                                  user_class=extracted_class)
                gp_ts_learner_feature2_1.update(pulled_arm_advertising, total_reward)

            instantaneous_reward_ts[t] = total_reward
            regret = opt_reward - total_reward
            instantaneous_regret_ts[t] = regret

        # 7 and 8 Week Estimation of 4 Classes or Decision of ending the experiment with 4 Classes
        else:
            # UCB1 and GP-UCB learners
            if features[1] == 0 and features[0] == 1:
                pulled_arm_pricing = ucb1_learner_c1.pull_arm()
                bernoulli_reward = env.round_step4(pulled_arm_pricing, extracted_class)
                ucb1_learner_c1.update(pulled_arm_pricing, bernoulli_reward * env.prices[pulled_arm_pricing])
                pulled_arm_advertising = gp_ucb_learner_c1.pull_arm()
                total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising,
                                                  user_class=extracted_class)
                gp_ucb_learner_c1.update(pulled_arm_advertising, total_reward)

            elif features[1] == 0 and features[0] == 0:
                pulled_arm_pricing = ucb1_learner_c2.pull_arm()
                bernoulli_reward = env.round_step4(pulled_arm_pricing, extracted_class)
                ucb1_learner_c2.update(pulled_arm_pricing, bernoulli_reward * env.prices[pulled_arm_pricing])
                pulled_arm_advertising = gp_ucb_learner_c2.pull_arm()
                total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising,
                                                  user_class=extracted_class)
                gp_ucb_learner_c2.update(pulled_arm_advertising, total_reward)

            elif features[1] == 1 and features[0] == 1:
                pulled_arm_pricing = ucb1_learner_c3.pull_arm()
                bernoulli_reward = env.round_step4(pulled_arm_pricing, extracted_class)
                ucb1_learner_c3.update(pulled_arm_pricing, bernoulli_reward * env.prices[pulled_arm_pricing])
                pulled_arm_advertising = gp_ucb_learner_c3.pull_arm()
                total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising,
                                                  user_class=extracted_class)
                gp_ucb_learner_c3.update(pulled_arm_advertising, total_reward)

            else:
                pulled_arm_pricing = ucb1_learner_c4.pull_arm()
                bernoulli_reward = env.round_step4(pulled_arm_pricing, extracted_class)
                ucb1_learner_c4.update(pulled_arm_pricing, bernoulli_reward * env.prices[pulled_arm_pricing])
                pulled_arm_advertising = gp_ucb_learner_c4.pull_arm()
                total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising,
                                                  user_class=extracted_class)
                gp_ucb_learner_c4.update(pulled_arm_advertising, total_reward)

            instantaneous_reward_ucb1[t] = total_reward
            regret = opt_reward - total_reward
            instantaneous_regret_ucb1[t] = regret

            # TS and GP-TS learners
            if features[1] == 0 and features[0] == 1:
                pulled_arm_pricing = ts_learner_c1.pull_arm(env.prices)
                pricing_reward = env.round_step4(pulled_arm_pricing, extracted_class)
                ts_learner_c1.update(pulled_arm_pricing, pricing_reward)
                pulled_arm_advertising = gp_ts_learner_c1.pull_arm()
                total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising,
                                                  user_class=extracted_class)
                gp_ts_learner_c1.update(pulled_arm_advertising, total_reward)
            elif features[1] == 0 and features[0] == 0:
                pulled_arm_pricing = ts_learner_c2.pull_arm(env.prices)
                pricing_reward = env.round_step4(pulled_arm_pricing, extracted_class)
                ts_learner_c2.update(pulled_arm_pricing, pricing_reward)
                pulled_arm_advertising = gp_ts_learner_c2.pull_arm()
                total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising,
                                                  user_class=extracted_class)
                gp_ts_learner_c2.update(pulled_arm_advertising, total_reward)
            elif features[1] == 1 and features[0] == 1:
                pulled_arm_pricing = ts_learner_c3.pull_arm(env.prices)
                pricing_reward = env.round_step4(pulled_arm_pricing, extracted_class)
                ts_learner_c3.update(pulled_arm_pricing, pricing_reward)
                pulled_arm_advertising = gp_ts_learner_c3.pull_arm()
                total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising,
                                                  user_class=extracted_class)
                gp_ts_learner_c3.update(pulled_arm_advertising, total_reward)
            else:
                pulled_arm_pricing = ts_learner_c4.pull_arm(env.prices)
                pricing_reward = env.round_step4(pulled_arm_pricing, extracted_class)
                ts_learner_c4.update(pulled_arm_pricing, pricing_reward)
                pulled_arm_advertising = gp_ts_learner_c4.pull_arm()
                total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising,
                                                  user_class=extracted_class)
                gp_ts_learner_c4.update(pulled_arm_advertising, total_reward)

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
    plot_statistics(inst_reward_clairvoyant, inst_regret_clairvoyant, 'Clairvoyant', 'Step 4.2 - Context generation')
    plot_statistics(inst_reward_ucb1, inst_regret_ucb1, 'UCB1 & GP-UCB', 'Step 4.2')
    plot_statistics(inst_reward_ts, inst_regret_ts, 'TS & GP-TS', 'Step 4.2')
    plt.tight_layout()
    plt.show()
