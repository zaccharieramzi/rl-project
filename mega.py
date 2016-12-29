import numpy as np
import random

from mega.routines import mega_routine

from plots import regret_plt

#  UNIVERSE PARAMETERS
n_users = 3
n_arms = 5
t_horizon = 100000

arm_means = [0.2, 0.3, 0.5, 0.8, 0.9]

# ALGORITHM PARAMETERS
params = {
    'c': 0.1,
    'd': 0.05,
    'alpha': 0.5,
    'beta': 0.8,
    'persistence_proba_init': 0.6
}


rewards, collisions = mega_routine(n_users, params, n_arms, t_horizon,
                                   arm_means, alg='eps')


# regret curve
print(collisions.sum())

chunk = t_horizon // 10
for i in range(10):
    print("{:d} iterations, {:d} collisions".format(
        chunk, collisions[:, i*chunk:(i+1)*chunk].sum()))
regret_plt(arm_means, rewards)
print("average reward {:f}, optimal 2.2".format(rewards.sum(axis=0).mean()))
