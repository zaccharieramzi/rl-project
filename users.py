import numpy as np
import random


class SecondaryUser:
    """
    class defining the behaviour and stored statistic of a user
    """

    def __init__(self, n_arms, params):
        self.params = params
        self.persistence_proba = params['persistence_proba_init']
        # an arm i is available if t >= arms_availability[i]
        self.available_arms = np.zeros(n_arms)
        self.arms_rew = np.zeros(n_arms)
        self.draws = np.zeros(n_arms, dtype='int')

        self.collided = False
        self.previous_arm = -1
        self.arm = -1

    def decision(self, t):
        '''
        return the arm choice of a user, if user changes arm we reset its
        *persistence_probability*
        Args :
               - t (int) timestep of the algorithm, necessary to know which
               arms are available
        '''
        self.previous_arm = self.arm
        # compute parameter for greedy algorithm
        n_arms = self.available_arms.shape[0]
        eps = min(1,
                  self.params['c'] * (n_arms ** 2) / (
                      self.params['d']**2 * (n_arms - 1) * t))
        # explore with proba eps
        if random.random() <= eps:
            # choose uniformely between available arms
            if (self.available_arms < t).sum() >= 1:
                self.arm = np.random.choice(
                    np.where(self.available_arms < t)[0], size=1)[0]
            else:
                # no available arms : no arms chosen.
                self.arm = -1
        else:
            # exploit !

            # best available empirical average of available arms
            available_arms = self.available_arms < t
            empirical_m = self.arms_rew / np.maximum(self.draws, 1)
            self.arm = np.argmax(available_arms * empirical_m)

        # if an arm has changed reset persistence proba
        if self.arm != self.previous_arm:
            self.persistence_proba = self.params['persistence_proba_init']
        return self.arm

    def draw_from_arm(self, arm):
        '''
        the user draws from the chosen arm and updates its statistics
        Args :
               - arm (object) arm to draw from
        Outputs : reward (int) the actual reward
        '''
        reward = arm.draw()
        self.arms_rew[self.arm] += reward
        self.draws[self.arm] += 1
        return reward
