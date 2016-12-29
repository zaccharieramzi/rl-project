import matplotlib.pyplot as plt
import numpy as np


def regret_plt(means, rewards):
    '''
    plot the regret curve through iterations
    Args :
           - means (list or vector) arms means
           - rewards (np array : n_users, t_horizon) rewards for each user
           at every timesteps
    '''
    n_users = rewards.shape[0]
    t_horizon = rewards.shape[1]

    n_best_arms = np.sort(np.array(means))[-n_users:]

    regret = np.cumsum(n_best_arms.sum() - rewards.sum(axis=0))
    plt.plot(range(t_horizon), regret, linewidth=2)
    plt.legend("regret")
    plt.show()
