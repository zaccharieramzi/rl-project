import numpy as np
from arms import ArmBernoulli
import random
import matplotlib.pyplot as plt
import collections
import math


# UNIVERSE PARAMETERS
n_users = 3
n_arms = 5
t_horizon = 10000

arm_means = [0.2, 0.3, 0.5, 0.8, 0.9]

arms = list()
for i in range(n_arms):
    arms.append(ArmBernoulli(arm_means[i]))

# ALGORITHM PARAMETERS


class SecondaryUser:
    ''' class used for user behaviour
    '''

    def __init__(self, n_arms, n_users, t_horizon):
        self.n_arms = n_arms
        self.n_users = n_users
        self.rewards = np.zeros((n_arms, t_horizon))
        self.draws = np.zeros(n_arms, dtype='int')
        self.arm_id = -1
        self.rank_to_consider = 0

    def decision(self, t, alg='ucb'):
        if alg == 'ucb':
            return self.decision_ucb(t)
        elif alg == 'ts':
            return self.decision_ts(t)

    def decision_ucb(self, t):
        ''' Choses the arm to draw at each time step t.
            Args:
                - t (int): the time step.
            Output:
                - int: the index of the arm chosen by this user.
        '''
        if np.sum(self.draws > 0) < self.n_arms:
            # in this case we are still in initialization: we want all arms to
            # have been drawn at least once.
            return (t)
        else:
            # in this case we are in the main loop
            ucb_stat = np.sum(self.rewards, axis=1) / self.draws +\
                np.sqrt(math.log(t) / self.draws)
            arms_sorted = np.argsort(ucb_stat)[::-1][:n_users]
            return arms_sorted[self.rank_to_consider]

    def decision_ts(self, t):
        ''' Choses the arm to draw at each time step t, following TS.
            Args:
                - t (int): the time step.
            Output:
                - int: the index of the arm chosen by this user.
        '''
        betas = np.zeros(self.n_arms)
        for arm_id in range(self.n_arms):
            betas[arm_id] = np.random.beta(
                np.sum(self.rewards[arm_id, :]) + 1,
                self.draws[arm_id] - np.sum(self.rewards[arm_id, :]) + 1)
        arms_sorted = np.argsort(betas)
        arms_sorted = arms_sorted[::-1]
        return arms_sorted[self.rank_to_consider]

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


def rho_rand_routine(n_users, n_arms, t_horizon, arm_means, alg='ucb',
                     plot=False):
    ''' Apply rho_rand avoidance strategy to a pb with t_horizon time steps.
        Args:
            - n_users (int): number of users.
            - n_arms (int): number of arms.
            - t_horizon (int): time steps.
            - alg (str): algorithm decision. 'ucb' or 'ts'
            - plot (Bool): True if plot is needed
        Output:
            - total_rewards (ndarray): total reward at each time step.
    '''
    users = [SecondaryUser(n_arms, n_users, t_horizon) for i in range(n_users)]
    total_rewards = np.zeros((t_horizon, 1))
    for t in range(t_horizon):
        # initialization
        if t < n_arms:
            choices = [user.decision(t) for user in users]
            for (user_id, choice) in enumerate(choices):
                arm = arms[choice]
                user = users[user_id]
                user.arm_id = choice
                user.draw_from_arm(arm, t)
        # main loop
        else:
            choices = [user.decision(t) for user in users]
            choice_count = collections.Counter(choices)
            # watch for collisions and update 'rank_to_consider'
            collisioned_users_id = (user_id for (user_id, choice)
                                    in enumerate(choices)
                                    if choice_count[choice] > 1)
            for user_id in collisioned_users_id:
                users[user_id].rank_to_consider = random.randrange(n_users)
            # draw the arms that have to be drawn (selected only ones)
            arms_to_draw = [choice for choice
                            in choices if choice_count[choice] == 1]
            for (user_id, choice) in enumerate(choices):
                if choice in arms_to_draw:
                    arm = arms[choice]
                    user = users[user_id]
                    user.arm_id = choice
                    reward = user.draw_from_arm(arm, t)
                    total_rewards[t] += reward
    if plot:
        best_arms = np.sort(np.array(arm_means))[-n_users:]
        regret = np.cumsum(best_arms.sum() - total_rewards)
        plt.plot(range(t_horizon), regret, linewidth=2)
        plt.ylabel("Regret")
        plt.xlabel("Time")
        plt.legend("Regret function of time for TDFS using {0}".format(alg))
        plt.show()
    return total_rewards
