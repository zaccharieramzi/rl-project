import matplotlib.pyplot as plt
import numpy as np


def regret_plt(best_arms_mean, rewards, upper_bound=None):
    '''
    plot the regret curve through iterations
    Args :
           - best_arms_mean (list or vector): means of the best arms considered
           - rewards (np array : t_horizon): total rewards at every timestep
           - upper_bound (np array : t_horizon): upper bound for the regret
           at every timestep
    '''
    regret = np.cumsum(best_arms_mean.sum() - rewards)
    plt.plot(regret, linewidth=2, label="Regret")
    if upper_bound:
        plt.plot(upper_bound, "r--", label="Upper Bound")
    plt.ylabel("Regret")
    plt.xlabel("Time")
    plt.legend(loc=2)
    plt.show()
