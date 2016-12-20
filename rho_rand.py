import numpy as np
import arms
from random import randrange

# implementation of the bandit environment
arm_means = [0.2, 0.3, 0.5, 0.8, 0.9]
n_arms = len(arm_means)

mab = list()
for i in range(n_arms):
    mab.append(ArmBernoulli(arm_means[i]))


def rho_rand_strategy(mab, n_users):
    ''' Plot the regret in function of the timee for a multi-players
            bandit problem
        Args:
            - arms (list): list of the used arms
            - n_users (int): number of users
        Output:
            - cumulative regret over time
    '''
    n_arms = len(mab)

    # Initialisation
    # sample-mean availability of channel i, as sensed by user j
    sample_mean = np.zeros(n_arms, n_users)

    # number of times user j draws arm i
    drawn_arms = np.ones(n_arms, n_users)

    # bandit algorithm statistic (here UCBà
    decision_statistic = np.zeros(n_arms, n_users)

    # 1 if collision happened for player j with arm i
    collision_matrix = zeros(n_arms, n_users)

    # which ranked arm is currently selected by player j
    current_arm_selection = np.arange(n_arms)

    # ranking of the favorites arms per player.
    # if favorite_arm(i, j) = x, x is the i^th favorite arm for player j
    favorite_arm = np.zeros(n_users, n_users)

    for user in range(n_users):
        for i in range(n_arms):
            sample_mean[i, user] = mab[i].draw()

        favorite_arm[:, user] = np.argsort(sample_mean[:, user])[::-1]
        decision_statistic =  # TODO:  update

    # loop until t_horizon is achieved
    for t in range(t_horizon):
        for user in range(n_users):
            if np.count_nonzero(
                                current_arm_selection ==
                                current_arm_selection[user]) != 0:
                collision_matrix[current_arm_selection[user], user] = 1

            if collision_matrix(current_arm_selection[user], user) == 1:
                current_arm_selection[user] = random.randrange(n_users)
            else:
                drawn_arms =  # TODO: : update
                sample_mean[
                            current_arm_selection[user],
                            user
                            ] = mab[current_arm_selection[user]].draw()
                decision_statistic =  # TODO: : update
                favorite_arm =  # TODO: : update
                regret[t] =  # TODO: : update
