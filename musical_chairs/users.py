import random
import numpy as np


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
