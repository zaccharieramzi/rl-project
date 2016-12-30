import numpy as np
import random
import matplotlib.pyplot as plt
import collections

from arms import ArmBernoulli


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
