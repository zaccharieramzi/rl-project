import numpy as np
from arms import ArmBernoulli
import random
import matplotlib.pyplot as plt
import collections
import math


class SecondaryUser:
    ''' class used for user behaviour
    '''

    def __init__(self, n_arms, n_users, t_horizon):
        self.n_arms = n_arms
        self.n_users = n_users
        self.rewards = np.zeros((n_arms, t_horizon))
        self.draws = np.zeros(n_arms, dtype='int')
        self.top_arms = np.zeros(n_users, dtype='int')
        self.fixed_on_arm = -1
        self.arm_id = -1

    def draw_from_arm(self, arm, t):
        ''' The user draws from the chosen arm and updates its statistics
                Args :
                        - arm (object) arm to draw from
                Outputs : reward (int) the actual reward
        '''
        reward = arm.draw()
        self.draws[self.arm_id] += 1
        self.rewards[self.arm_id, t] = reward
        return reward

    def rank_arms(self):
        ''' Rank arms before phase 2.
            Output:
                - ndarray: n_users top arms.
        '''
        user_stat = np.sum(self.rewards, axis=1) / self.draws
        arms_sorted = np.argsort(user_stat)[::-1][:self.n_users]
        return arms_sorted


def mc_routine(n_users, params, n_arms, t_horizon, arm_means, plot=False):
    ''' Apply muscial chairs algorithm to a pb with t_horizon time steps.
        Args:
            - n_users (int): number of users.
            - n_arms (int): number of arms.
            - params (dict): t0 - exploring phase / t1 - exploiting phase
            - t_horizon (int): time steps.
            - plot (Bool): True if plot is needed
        Output:
            - total_rewards (ndarray): total reward at each time step.
    '''
    arms = list()
    for i in range(n_arms):
        arms.append(ArmBernoulli(arm_means[i]))
    users = [SecondaryUser(n_arms, n_users, t_horizon) for i in range(n_users)]
    total_rewards = np.zeros((t_horizon, 1))
    if t_horizon < params['t1']:
        return print("horizon must be at least t1")
    t_temp = 0
    for t in range(t_horizon):
        # phase 1: exploring the arms in order to rank them.
        if t_temp < params['t0']:
            choices = [random.randrange(n_arms) for user in users]
            choice_count = collections.Counter(choices)
            arms_to_draw = [choice for choice
                            in choices if choice_count[choice] == 1]
            for (user_id, choice) in enumerate(choices):
                if choice in arms_to_draw:
                    arm = arms[choice]
                    user = users[user_id]
                    user.arm_id = choice
                    reward = user.draw_from_arm(arm, t)
                    total_rewards[t] += reward
            t_temp += 1
        # rank arms when we reach t0 steps
        if t_temp == params['t0']:
            for user_id in range(n_users):
                users[user_id].top_arms = users[user_id].rank_arms()
        # phase 2 once all the players are fixed on an arm.
        if t_temp < params['t1'] and t_temp >= params['t0']:
            for (idx, user) in enumerate(users):
                if user.fixed_on_arm == -1:
                    choices[idx] = user.top_arms[random.randrange(n_users)]
                else:
                    choices[idx] = user.arm_id
            # draw the arms that have to be drawn (selected only ones)
            choice_count = collections.Counter(choices)
            arms_to_draw = [choice for choice
                            in choices if choice_count[choice] == 1]
            for (user_id, choice) in enumerate(choices):
                if choice in arms_to_draw:
                    arm = arms[choice]
                    user = users[user_id]
                    user.arm_id = choice
                    if user.fixed_on_arm == -1:
                        user.fixed_on_arm = +1
                    reward = user.draw_from_arm(arm, t)
                    total_rewards[t] += reward
            t_temp += 1
        if t_temp == params['t1']:
            t_temp = 0
            for user_id in range(n_users):
                users[user_id].fixed_on_arm = -1
    if plot:
        best_arms = np.sort(np.array(arm_means))[-n_users:]
        regret = np.cumsum(best_arms.sum() - total_rewards)
        plt.plot(range(t_horizon), regret, linewidth=2)
        plt.ylabel("Regret")
        plt.xlabel("Time")
        plt.legend("Regret function of time for Musical chairs")
        plt.show()
    return total_rewards
