import math
import numpy as np
import random


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
