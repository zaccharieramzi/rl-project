import numpy as np
import random

from arms import ArmBernoulli


#  UNIVERSE PARAMETERS
n_users = 3
n_arms = 5
t_horizon = 100

arm_means = [0.2, 0.3, 0.5, 0.8, 0.9]

arms = list()
for i in range(n_arms):
    arms.append(ArmBernoulli(arm_means[i]))

# ALGORITHM PARAMETERS


class SecondaryUser:
    ''' class used for user behaviour
    '''

    def __init__(self):
        self.n_arms = n_arms
        self.offset = random.randint(0, n_players-1)
        self.rewards = np.zeros((n_arms, t_horizon))
        self.draws = np.zeros((n_arms, 1))

    def decision(self, t):
        ''' Choses the arm to draw at each time step t.
            Args:
                - t (int): the time step.
            Output:
                - int: the index of the arm chosen by this user.
        '''
        if np.sum(self.draws > 0) < n_arms:
            # in this case we are still in initialization: we want all arms to
            # have been drawn at least once.
            return (t % n_arms) - self.offset
        else:
            # in this case we are in the main loop
            top_arm_to_consider = (t % n_players) - self.offset
            ucb_stat = np.sum(self.rewards, axis=1) / self.draws +\
                np.sqrt(log(t) / self.draws)
            best_arms = np.argsort(ucb_stat)
            return best_arms(top_arm_to_consider)
