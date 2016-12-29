import collections
import math
import random

import matplotlib.pyplot as plt
import numpy as np

from arms import ArmBernoulli
from .users import SecondaryUser


def tdfs_routine(n_users, n_arms, t_horizon, arm_means, alg='ucb', plot=False):
    arms = list()
    for i in range(n_arms):
        arms.append(ArmBernoulli(arm_means[i]))
    users = [SecondaryUser(n_arms, n_users, t_horizon) for i in range(n_users)]
    total_rewards = np.zeros((t_horizon, 1))
    for t in range(t_horizon):
        choices = [user.decision(t, alg=alg) for user in users]
        choice_count = collections.Counter(choices)
        collisioned_users_id = (user_id for (user_id, choice)
                                in enumerate(choices)
                                if choice_count[choice] > 1)
        for user_id in collisioned_users_id:
            users[user_id].collided_in_subsequence = True
        arms_to_draw = [choice for choice in choices
                        if choice_count[choice] == 1]
        for (user_id, choice) in enumerate(choices):
            if choice in arms_to_draw:
                arm = arms[choice]
                user = users[user_id]
                reward = arm.draw()
                user.draws[choice] += 1
                user.rewards[choice, t] = reward
                total_rewards[t] += reward
        if (t == n_arms) or ((t > n_arms) and ((t - n_arms) % n_users == 0)):
            # We are at the end of a subsequence corresponding to
            # initialization or classic.
            # Therefore, we must correct the offsets.
            for user in users:
                if user.collided_in_subsequence:
                    user.offset = random.randint(0, user.n_users - 1)
                    user.collided_in_subsequence = False
    if plot:
        best_arms = np.sort(np.array(arm_means))[-n_users:]
        regret = np.cumsum(best_arms.sum() - total_rewards)
        plt.plot(range(t_horizon), regret, linewidth=2)
        plt.ylabel("Regret")
        plt.xlabel("Time")
        plt.legend("Regret function of time for TDFS using {0}".format(alg))
        plt.show()
    return total_rewards


def kl_divergence_bernoulli(p, q):
    return p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q))


def x_k(arm_means, k):
    ordered_arm_means = np.sort(arm_means)
    ordered_arm_means = ordered_arm_means[::-1]
    ordered_arm_means = ordered_arm_means[:k]
    return sum(
        (sum(
            (1 / kl_divergence_bernoulli(arm_mean_smaller, arm_mean)
             for arm_mean_smaller in arm_means
             if arm_mean_smaller < arm_mean))
         for arm_mean in ordered_arm_means))


def log_upper_bound(n_users, arm_means):
    ordered_arm_means = np.sort(arm_means)
    ordered_arm_means = ordered_arm_means[::-1]
    first_sum = sum(
        (sum(
            (x_k(arm_means, k) * ordered_arm_means[i]
             for k in range(n_users)))
         for i in range(n_users)))
    second_sum = sum(
        (arm_mean * max(1/kl_divergence_bernoulli(
            arm_mean, ordered_arm_means[n_users]), 0)
         for arm_mean in arm_means if arm_mean < ordered_arm_means[n_users]))
    return n_users * (first_sum-second_sum)
