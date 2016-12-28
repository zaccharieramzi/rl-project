import collections
import math
import random

import matplotlib.pyplot as plt
import numpy as np

from arms import ArmBernoulli


class SecondaryUser:
    ''' class used for user behaviour
    '''

    def __init__(self, n_arms, n_users, t_horizon):
        self.n_arms = n_arms
        self.n_users = n_users
        self.offset = random.randint(0, n_users - 1)
        self.rewards = np.zeros((n_arms, t_horizon))
        self.draws = np.zeros(n_arms)
        self.collided_in_subsequence = False

    def decision(self, t, alg='ucb'):
        if alg == 'ucb':
            return self.decision_ucb(t)
        elif alg == 'ts':
            return self.decision_ts(t)

    def decision_ucb(self, t):
        ''' Choses the arm to draw at each time step t, following UCB.
            Args:
                - t (int): the time step.
            Output:
                - int: the index of the arm chosen by this user.
        '''
        if np.sum(self.draws > 0) < self.n_arms:
            # in this case we are still in initialization: we want all arms to
            # have been drawn at least once.
            return (t + self.offset) % self.n_arms
        else:
            # in this case we are in the main loop
            top_arm_to_consider = (t - self.n_arms + self.offset) %\
                self.n_users
            ucb_stat = np.sum(self.rewards, axis=1) / self.draws +\
                np.sqrt(math.log(t) / self.draws)
            arms_sorted = np.argsort(ucb_stat)
            arms_sorted = arms_sorted[::-1]
            return arms_sorted[top_arm_to_consider]

    def decision_ts(self, t):
        ''' Choses the arm to draw at each time step t, following TS.
            Args:
                - t (int): the time step.
            Output:
                - int: the index of the arm chosen by this user.
        '''
        betas = np.zeros(self.n_arms)
        top_arm_to_consider = (t + self.offset) % self.n_users
        for arm_id in range(self.n_arms):
            betas[arm_id] = np.random.beta(
                np.sum(self.rewards[arm_id, :]) + 1,
                self.draws[arm_id] - np.sum(self.rewards[arm_id, :]) + 1)
        arms_sorted = np.argsort(betas)
        arms_sorted = arms_sorted[::-1]
        return arms_sorted[top_arm_to_consider]


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
