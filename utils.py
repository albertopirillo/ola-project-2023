import numpy as np
import matplotlib.pyplot as plt


def compute_statistics(instantaneous_values: np.ndarray[float]) ->\
        tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Compute the mean and the standard deviation for both the instantaneous and the cumulative values
    :param instantaneous_values: a 2D array of shape (num_experiments, num_time_instants)
    :return: a tuple of 4 1D arrays of shape (num_time_instants) containing the mean and std over the experiments of
     the instantaneous values and of the cumulative values
    """
    # Compute mean and std of the instantaneous values
    inst_mean = np.mean(instantaneous_values, axis=0)
    inst_std = np.std(instantaneous_values, axis=0)

    # Compute mean and std of the cumulative values
    cumulative_values = np.cumsum(instantaneous_values, axis=1)
    cumulative_mean = np.mean(cumulative_values, axis=0)
    cumulative_std = np.std(cumulative_values, axis=0)

    return inst_mean, inst_std, cumulative_mean, cumulative_std


def plot_with_std(figure_id: int, y_label: str, curve_label: str, 
                  values_mean: np.ndarray[float], values_std: np.ndarray[float]) -> None:
    """
    Plot the mean of the values with the corresponding standard deviation
    :param figure_id: the id of the figure
    :param y_label: the label of the y axis
    :param curve_label: the label to show on the legend
    :param values_mean: a 1D array of shape (num_time_instants) containing the mean of the values
    :param values_std: a 1D array of shape (num_time_instants) containing the std of the values
    :return:
    """
    plt.figure(figure_id)
    plt.xlabel('t')
    plt.ylabel(y_label)
    plt.plot(values_mean, label=curve_label)
    plt.fill_between(range(len(values_mean)), values_mean - values_std, values_mean + values_std, alpha=0.2)
    plt.legend()


def plot_statistics(instantaneous_rewards: np.ndarray[float],
                    instantaneous_regrets: np.ndarray[float], label: str) -> None:
    """
    Plot the mean of the instantaneous and cumulative rewards and regrets with the corresponding standard deviation
    Once all plots have been generated, call plt.show() to display them
    :param instantaneous_rewards: a 2D array of shape (num_experiments, num_time_instants)
    :param instantaneous_regrets: a 2D array of shape (num_experiments, num_time_instants)
    :param label: the label to show on the legend
    :return:
    """
    i_reward_mean, i_reward_std, c_reward_mean, c_reward_std = compute_statistics(instantaneous_rewards)
    i_regret_mean, i_regret_std, c_regret_mean, c_regret_std = compute_statistics(instantaneous_regrets)
    plot_with_std(0, 'Instantaneous reward', label, i_reward_mean, i_reward_std)
    plot_with_std(1, 'Instantaneous regret', label, i_regret_mean, i_regret_std)
    plot_with_std(2, 'Cumulative reward', label, c_reward_mean, c_reward_std)
    plot_with_std(3, 'Cumulative regret', label, c_regret_mean, c_regret_std)
