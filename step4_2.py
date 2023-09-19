
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from tqdm.contrib.concurrent import process_map

from environments.ClairvoyantAlgorithm import ClairvoyantAlgorithm
from environments.Environment import Environment
from learners.GPTSLearner import GPTSLearner
from learners.GPUCBLearner import GPUCBLearner
from learners.Learner import Learner
from learners.TSLearner import TSLearner
from learners.UCB1Learner import UCB1Learner
from utils import plot_statistics, hoeffding_bound, plot_contexts

import warnings
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

# Simulation parameters
T = 365
n_experiments = 250
hoeffding_bound_confidence = 0.8


class ChoiceOfContext(Enum):
    NO_CHOICE = auto(),
    ONE_CLASS = auto(),
    TWO_CLASSES_F1 = auto(),
    TWO_CLASSES_F2 = auto(),
    FOUR_CLASSES = auto(),


def extract_class(env: Environment, features_count: list[int, int]) -> tuple[int, list[int, int]]:
    extracted_class: int = np.random.choice(env.num_classes, p=env.class_probabilities)
    features: list[int, int] = [0, 0]
    match extracted_class:
        case 0:
            features[0] = 0
            features[1] = 0
        case 1:
            choice = int(np.random.choice(np.arange(2), size=1))
            features[0] = choice
            features[1] = 1 - features[0]
            features_count[0] += choice
            features_count[1] += 1 - features[0]
        case 2:
            features[0] = 1
            features[1] = 1
            features_count[0] += 1
            features_count[1] += 1

    return extracted_class, features


def select_learner_from_context(context: ChoiceOfContext, features: list[int, int],
                                learner_list: list[Learner]) -> Learner:
    match context:
        case ChoiceOfContext.ONE_CLASS:
            # ONE_CLASS learner is always updated
            learner = learner_list[0]
        case ChoiceOfContext.TWO_CLASSES_F1:
            # TWO_CLASSES learners: update the learner that matches the features
            learner = learner_list[features[0]]
        case ChoiceOfContext.TWO_CLASSES_F2:
            learner = learner_list[features[1] + 2]
        case ChoiceOfContext.FOUR_CLASSES:
            # FOUR_CLASSES learners: update the learner that matches the features
            learner = learner_list[features[0] + features[1] * 2]
        case _:
            raise ValueError('Invalid context')

    return learner


def update_pricing_learner(env: Environment, features: list[int, int], extracted_class: int, context: ChoiceOfContext,
                           learner_list: list[Learner]) -> int:
    learner = select_learner_from_context(context, features, learner_list)
    pulled_arm_pricing = learner.pull_arm(env.prices)
    bernoulli_reward = env.round_step4(pulled_arm_pricing, extracted_class)

    if isinstance(learner, UCB1Learner):
        learner.update(pulled_arm_pricing, float(bernoulli_reward * env.prices[pulled_arm_pricing]))
    else:
        learner.update(pulled_arm_pricing, bernoulli_reward)

    return pulled_arm_pricing


def update_advertising_learner(env: Environment, features: list[int, int], extracted_class: int,
                               context: ChoiceOfContext, learner_list: list[Learner], pulled_arm_pricing: int) -> int:
    learner = select_learner_from_context(context, features, learner_list)
    pulled_arm_advertising = learner.pull_arm()
    total_reward = env.compute_reward(pulled_arm_pricing, pulled_arm_advertising, user_class=extracted_class)
    learner.update(pulled_arm_advertising, total_reward)
    return pulled_arm_advertising


def compute_feature_prob(count: int, t: int) -> tuple[float, float]:
    positive_count = count / t
    return 1 - positive_count, positive_count


def compute_context_value(context: ChoiceOfContext, learner_list: list[Learner], features_count: list[int, int],
                          confidence: float, t: int) -> float:
    match context:
        case ChoiceOfContext.ONE_CLASS:
            mu = learner_list[0].get_best_expected_value()
            mu_bound = hoeffding_bound(mu, confidence, t)
            return mu_bound

        case ChoiceOfContext.TWO_CLASSES_F1:
            p1, p2 = compute_feature_prob(features_count[0], t)
            p1_bound, p2_bound = hoeffding_bound(p1, confidence, t), hoeffding_bound(p2, confidence, t)

            mu1, mu2 = learner_list[0].get_best_expected_value(), learner_list[1].get_best_expected_value()
            mu1_bound, mu2_bound = hoeffding_bound(mu1, confidence, t), hoeffding_bound(mu2, confidence, t)
            return p1_bound * mu1_bound + p2_bound * mu2_bound

        case ChoiceOfContext.TWO_CLASSES_F2:
            p1, p2 = compute_feature_prob(features_count[1], t)
            p1_bound, p2_bound = hoeffding_bound(p1, confidence, t), hoeffding_bound(p2, confidence, t)

            mu1, mu2 = learner_list[2].get_best_expected_value(), learner_list[3].get_best_expected_value()
            mu1_bound, mu2_bound = hoeffding_bound(mu1, confidence, t), hoeffding_bound(mu2, confidence, t)
            return p1_bound * mu1_bound + p2_bound * mu2_bound

        case ChoiceOfContext.FOUR_CLASSES:
            p1, p2 = compute_feature_prob(features_count[0], t)
            p3, p4 = compute_feature_prob(features_count[1], t)
            p1_bound, p2_bound, p3_bound, p4_bound = hoeffding_bound(p1, confidence, t), \
                hoeffding_bound(p2, confidence, t), hoeffding_bound(p3, confidence, t), \
                hoeffding_bound(p4, confidence, t)

            mu1, mu2, mu3, mu4 = learner_list[0].get_best_expected_value(), learner_list[1].get_best_expected_value(), \
                learner_list[2].get_best_expected_value(), learner_list[3].get_best_expected_value()
            mu1_bound, mu2_bound, mu3_bound, mu4_bound = hoeffding_bound(mu1, confidence, t), \
                hoeffding_bound(mu2, confidence, t), hoeffding_bound(mu3, confidence, t), \
                hoeffding_bound(mu4, confidence, t)

            return p1_bound * mu1_bound + p2_bound * mu2_bound + p3_bound * mu3_bound + p4_bound * mu4_bound


def run_experiment(_):
    # For every experiment, we define new environment and learners
    env = Environment.from_json('data/environment_step4.json')
    # No information on pricing nor advertising
    clairvoyant = ClairvoyantAlgorithm(env)

    # Learners
    # Context With a Single Class
    ucb1_learner_1class = [UCB1Learner(len(env.prices))]
    ts_learner_1class = [TSLearner(len(env.prices))]
    gp_ucb_learner_1class = [GPUCBLearner(len(env.bids), env.bids)]
    gp_ts_learner_1class = [GPTSLearner(len(env.bids), env.bids)]

    # Context With Two Classes (the first two learners are split on feature 1, the others on feature 2)
    ucb1_learner_2classes = [UCB1Learner(len(env.prices)), UCB1Learner(len(env.prices)), UCB1Learner(len(env.prices)),
                             UCB1Learner(len(env.prices))]
    ts_learner_2classes = [TSLearner(len(env.prices)), TSLearner(len(env.prices)), TSLearner(len(env.prices)),
                           TSLearner(len(env.prices))]
    gp_ucb_learner_2classes = [GPUCBLearner(len(env.bids), env.bids), GPUCBLearner(len(env.bids), env.bids),
                               GPUCBLearner(len(env.bids), env.bids), GPUCBLearner(len(env.bids), env.bids)]
    gp_ts_learner_2classes = [GPTSLearner(len(env.bids), env.bids), GPTSLearner(len(env.bids), env.bids),
                              GPTSLearner(len(env.bids), env.bids), GPTSLearner(len(env.bids), env.bids)]

    # Context With Four Classes
    ucb1_learner_4classes = [UCB1Learner(len(env.prices)), UCB1Learner(len(env.prices)), UCB1Learner(len(env.prices)),
                             UCB1Learner(len(env.prices))]
    ts_learner_4classes = [TSLearner(len(env.prices)), TSLearner(len(env.prices)), TSLearner(len(env.prices)),
                           TSLearner(len(env.prices))]
    gp_ucb_learner_4classes = [GPUCBLearner(len(env.bids), env.bids), GPUCBLearner(len(env.bids), env.bids),
                               GPUCBLearner(len(env.bids), env.bids), GPUCBLearner(len(env.bids), env.bids)]
    gp_ts_learner_4classes = [GPTSLearner(len(env.bids), env.bids), GPTSLearner(len(env.bids), env.bids),
                              GPTSLearner(len(env.bids), env.bids), GPTSLearner(len(env.bids), env.bids)]

    # Data structures
    instantaneous_reward_clairvoyant = np.zeros(T)
    instantaneous_reward_ucb = np.zeros(T)
    instantaneous_reward_ts = np.zeros(T)

    instantaneous_regret_clairvoyant = np.zeros(T)
    instantaneous_regret_ucb = np.zeros(T)
    instantaneous_regret_ts = np.zeros(T)

    pricing_context_ucb = np.zeros(T)
    pricing_context_ts = np.zeros(T)
    advertising_context_ucb = np.zeros(T)
    advertising_context_ts = np.zeros(T)

    # Data structures to keep track of the values of the features
    features_count = [0, 0]

    # Data structures to keep track of the best arms for every context
    pulled_arms_pricing_ucb = [0, 0, 0, 0]
    pulled_arms_advertising_ucb = [0, 0, 0, 0]
    pulled_arms_pricing_ts = [0, 0, 0, 0]
    pulled_arms_advertising_ts = [0, 0, 0, 0]

    # Initialize the context to be a single class
    best_context_pricing_ucb, best_context_pricing_ts = 0, 0
    best_context_advertising_ucb, best_context_advertising_ts = 0, 0

    for t in range(T):
        # Clairvoyant algorithm
        extracted_class, features = extract_class(env, features_count)
        opt_reward = clairvoyant.optimal_rewards[extracted_class]
        instantaneous_reward_clairvoyant[t] = opt_reward
        instantaneous_regret_clairvoyant[t] = 0

        # Update all the learners
        # ONE_CLASS learners
        pulled_arms_pricing_ucb[0] = update_pricing_learner(env, features, extracted_class, ChoiceOfContext.ONE_CLASS,
                                                            ucb1_learner_1class)
        pulled_arms_advertising_ucb[0] = update_advertising_learner(env, features, extracted_class,
                                                                    ChoiceOfContext.ONE_CLASS, gp_ucb_learner_1class,
                                                                    pulled_arms_pricing_ucb[0])
        pulled_arms_pricing_ts[0] = update_pricing_learner(env, features, extracted_class, ChoiceOfContext.ONE_CLASS,
                                                           ts_learner_1class)
        pulled_arms_advertising_ts[0] = update_advertising_learner(env, features, extracted_class,
                                                                   ChoiceOfContext.ONE_CLASS, gp_ts_learner_1class,
                                                                   pulled_arms_pricing_ts[0])

        # TWO_CLASSES learners
        pulled_arms_pricing_ucb[1] = update_pricing_learner(env, features, extracted_class,
                                                            ChoiceOfContext.TWO_CLASSES_F1, ucb1_learner_2classes)
        pulled_arms_advertising_ucb[1] = update_advertising_learner(env, features, extracted_class,
                                                                    ChoiceOfContext.TWO_CLASSES_F1,
                                                                    gp_ucb_learner_2classes, pulled_arms_pricing_ucb[1])
        pulled_arms_pricing_ts[1] = update_pricing_learner(env, features, extracted_class,
                                                           ChoiceOfContext.TWO_CLASSES_F1, ts_learner_2classes)
        pulled_arms_advertising_ts[1] = update_advertising_learner(env, features, extracted_class,
                                                                   ChoiceOfContext.TWO_CLASSES_F1,
                                                                   gp_ts_learner_2classes, pulled_arms_pricing_ts[1])

        pulled_arms_pricing_ucb[2] = update_pricing_learner(env, features, extracted_class,
                                                            ChoiceOfContext.TWO_CLASSES_F2, ucb1_learner_2classes)
        pulled_arms_advertising_ucb[2] = update_advertising_learner(env, features, extracted_class,
                                                                    ChoiceOfContext.TWO_CLASSES_F2,
                                                                    gp_ucb_learner_2classes, pulled_arms_pricing_ucb[2])
        pulled_arms_pricing_ts[2] = update_pricing_learner(env, features, extracted_class,
                                                           ChoiceOfContext.TWO_CLASSES_F2, ts_learner_2classes)
        pulled_arms_advertising_ts[2] = update_advertising_learner(env, features, extracted_class,
                                                                   ChoiceOfContext.TWO_CLASSES_F2,
                                                                   gp_ts_learner_2classes,
                                                                   pulled_arms_pricing_ts[2])

        # FOUR_CLASSES learners
        pulled_arms_pricing_ucb[3] = update_pricing_learner(env, features, extracted_class,
                                                            ChoiceOfContext.FOUR_CLASSES, ucb1_learner_4classes)
        pulled_arms_advertising_ucb[3] = update_advertising_learner(env, features, extracted_class,
                                                                    ChoiceOfContext.FOUR_CLASSES,
                                                                    gp_ucb_learner_4classes, pulled_arms_pricing_ucb[3])
        pulled_arms_pricing_ts[3] = update_pricing_learner(env, features, extracted_class, ChoiceOfContext.FOUR_CLASSES,
                                                           ts_learner_4classes)
        pulled_arms_advertising_ts[3] = update_advertising_learner(env, features, extracted_class,
                                                                   ChoiceOfContext.FOUR_CLASSES, gp_ts_learner_4classes,
                                                                   pulled_arms_pricing_ts[3])

        # Run context generation every two weeks
        if t != 0 and t % 14 == 0:
            # For every context, compute its value and select the best one
            # Pricing UCB
            one_class = compute_context_value(ChoiceOfContext.ONE_CLASS, ucb1_learner_1class, features_count,
                                              hoeffding_bound_confidence, t)
            two_classes_f1 = compute_context_value(ChoiceOfContext.TWO_CLASSES_F1, ucb1_learner_2classes,
                                                   features_count, hoeffding_bound_confidence, t)
            two_classes_f2 = compute_context_value(ChoiceOfContext.TWO_CLASSES_F2, ucb1_learner_2classes,
                                                   features_count, hoeffding_bound_confidence, t)
            four_classes = compute_context_value(ChoiceOfContext.FOUR_CLASSES, ucb1_learner_4classes, features_count,
                                                 hoeffding_bound_confidence, t)
            best_context_pricing_ucb = np.argmax([one_class, two_classes_f1, two_classes_f2, four_classes])

            # Pricing TS
            one_class = compute_context_value(ChoiceOfContext.ONE_CLASS, ts_learner_1class, features_count,
                                              hoeffding_bound_confidence, t)
            two_classes_f1 = compute_context_value(ChoiceOfContext.TWO_CLASSES_F1, ts_learner_2classes, features_count,
                                                   hoeffding_bound_confidence, t)
            two_classes_f2 = compute_context_value(ChoiceOfContext.TWO_CLASSES_F2, ts_learner_2classes, features_count,
                                                   hoeffding_bound_confidence, t)
            four_classes = compute_context_value(ChoiceOfContext.FOUR_CLASSES, ts_learner_4classes, features_count,
                                                 hoeffding_bound_confidence, t)
            best_context_pricing_ts = np.argmax([one_class, two_classes_f1, two_classes_f2, four_classes])

            # Advertising UCB
            one_class = compute_context_value(ChoiceOfContext.ONE_CLASS, gp_ucb_learner_1class, features_count,
                                              hoeffding_bound_confidence, t)
            two_classes_f1 = compute_context_value(ChoiceOfContext.TWO_CLASSES_F1, gp_ucb_learner_2classes,
                                                   features_count, hoeffding_bound_confidence, t)
            two_classes_f2 = compute_context_value(ChoiceOfContext.TWO_CLASSES_F2, gp_ucb_learner_2classes,
                                                   features_count, hoeffding_bound_confidence, t)
            four_classes = compute_context_value(ChoiceOfContext.FOUR_CLASSES, gp_ucb_learner_4classes, features_count,
                                                 hoeffding_bound_confidence, t)
            best_context_advertising_ucb = np.argmax([one_class, two_classes_f1, two_classes_f2, four_classes])

            # Advertising TS
            one_class = compute_context_value(ChoiceOfContext.ONE_CLASS, gp_ts_learner_1class, features_count,
                                              hoeffding_bound_confidence, t)
            two_classes_f1 = compute_context_value(ChoiceOfContext.TWO_CLASSES_F1, gp_ts_learner_2classes,
                                                   features_count, hoeffding_bound_confidence, t)
            two_classes_f2 = compute_context_value(ChoiceOfContext.TWO_CLASSES_F2, gp_ts_learner_2classes,
                                                   features_count, hoeffding_bound_confidence, t)
            four_classes = compute_context_value(ChoiceOfContext.FOUR_CLASSES, gp_ts_learner_4classes, features_count,
                                                 hoeffding_bound_confidence, t)
            best_context_advertising_ts = np.argmax([one_class, two_classes_f1, two_classes_f2, four_classes])

        # Compute the reward and the regret
        # UCB & GP-UCB
        best_arm_pricing_ucb = pulled_arms_pricing_ucb[best_context_pricing_ucb]
        best_arm_advertising_ucb = pulled_arms_advertising_ucb[best_context_advertising_ucb]
        total_reward = env.compute_reward(best_arm_pricing_ucb, best_arm_advertising_ucb, user_class=extracted_class)

        instantaneous_reward_ucb[t] = total_reward
        regret = opt_reward - total_reward
        instantaneous_regret_ucb[t] = regret

        # TS & GP-TS
        best_arm_pricing_ts = pulled_arms_pricing_ts[best_context_pricing_ts]
        best_arm_advertising_ts = pulled_arms_advertising_ts[best_context_advertising_ts]
        total_reward = env.compute_reward(best_arm_pricing_ts, best_arm_advertising_ts, user_class=extracted_class)

        instantaneous_reward_ts[t] = total_reward
        regret = opt_reward - total_reward
        instantaneous_regret_ts[t] = regret

        # Keep track of the contexts
        pricing_context_ucb[t] = best_context_pricing_ucb
        pricing_context_ts[t] = best_context_pricing_ts
        advertising_context_ucb[t] = best_context_advertising_ucb
        advertising_context_ts[t] = best_context_advertising_ts

    return instantaneous_reward_clairvoyant, instantaneous_reward_ucb, instantaneous_reward_ts, \
        instantaneous_regret_clairvoyant, instantaneous_regret_ucb, instantaneous_regret_ts, pricing_context_ucb, \
        pricing_context_ts, advertising_context_ucb, advertising_context_ts


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

    pricing_context_history_ucb = results_array[:, 6, :]
    pricing_context_history_ts = results_array[:, 7, :]
    advertising_context_history_ucb = results_array[:, 8, :]
    advertising_context_history_ts = results_array[:, 9, :]

    # Generate plots of the mean and standard deviation of the results
    plot_statistics(inst_reward_clairvoyant, inst_regret_clairvoyant, 'Clairvoyant', 'Step 4.2 - Context generation')
    plot_statistics(inst_reward_ucb1, inst_regret_ucb1, 'UCB1 & GP-UCB', 'Step 4.2 - Context generation')
    plot_statistics(inst_reward_ts, inst_regret_ts, 'TS & GP-TS', 'Step 4.2 - Context generation')

    # Generate a plots to keep track of the evolution of the context generation
    plot_contexts([pricing_context_history_ucb, pricing_context_history_ts, advertising_context_history_ucb,
                   advertising_context_history_ts], ['Pricing UCB', 'Pricing TS', 'Advertising UCB', 'Advertising TS'],
                  title='Step 4.2 - Context generation')

    plt.tight_layout()
    plt.show()
