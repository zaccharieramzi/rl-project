import numpy as np
from arms import ArmBernoulli
import random
import matplotlib.pyplot as plt
import collections


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

    def __init__(self, n_arms, n_users):
        self.n_arms = n_arms
        self.n_users = n_users
        self.rewards = np.zeros((n_arms, t_horizon))
        self.draws = np.zeros(n_arms, dtype='int')
        self.arms_rew = np.zeros(n_arms)
        self.arm_id = -1
        self.rank_to_consider = 0

    def decision(self, t):
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
                np.sqrt(np.log(t) / self.draws)
            arms_sorted = np.argsort(ucb_stat)[::-1][:n_users]
            return arms_sorted[self.rank_to_consider]

    def draw_from_arm(self, arm):
        ''' The user draws from the chosen arm and updates its statistics
                Args :
                        - arm (object) arm to draw from
                Outputs : reward (int) the actual reward
        '''
        reward = arm.draw()
        self.arms_rew[self.arm_id] += reward
        self.draws[self.arm_id] += 1
        self.rewards[self.arm_id, t] = reward
        return reward


users = [SecondaryUser(n_arms, n_users) for i in range(n_users)]
total_rewards = np.zeros((t_horizon, 1))
for t in range(t_horizon):
    if t < n_arms:
        # initialization
        choices = [user.decision(t) for user in users]
        for (user_id, choice) in enumerate(choices):
            arm = arms[choice]
            user = users[user_id]
            user.arm_id = choice
            user.draw_from_arm(arm)
    else:
        # main loop
        choices = [user.decision(t) for user in users]
        choice_count = collections.Counter(choices)
        # watch for collisions and update 'arm_to_consider'
        collisioned_users_id = (user_id for (user_id, choice)
                                in enumerate(choices)
                                if choice_count[choice] > 1)
        for user_id in collisioned_users_id:
            users[user_id].rank_to_consider = random.randrange(n_users)
        # draw the arms that have to be drawn (selected ones only)
        arms_to_draw = [choice for choice
                        in choices if choice_count[choice] == 1]
        for (user_id, choice) in enumerate(choices):
            if choice in arms_to_draw:
                arm = arms[choice]
                user = users[user_id]
                user.arm_id = choice
                reward = user.draw_from_arm(arm)
                total_rewards[t] += reward

best_arms = np.sort(np.array(arm_means))[-n_users:]
regret = np.cumsum(best_arms.sum() - total_rewards)
plt.plot(range(t_horizon), regret, linewidth=2)
plt.legend("regret")
plt.show()
