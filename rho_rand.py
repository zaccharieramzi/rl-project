import numpy as np
from arms import ArmBernoulli
import random
from math import sqrt, log
import matplotlib.pyplot as plt


# implementation of the bandit environment
means = [0.2, 0.3, 0.5, 0.8, 0.9]
n_arms = len(means)

mab = list()
for i in range(n_arms):
    mab.append(ArmBernoulli(means[i]))

n_users = 1
t_horizon = 10000


def rho_rand_strategy(mab, n_users, t_horizon):
    ''' Plot the regret in function of time for a multi-players bandit problem
        Args:
            - arms (list): list of the used arms
            - n_users (int): number of users
            - t_horizon (int): rounds for the bandit game
        Output:
            - cumulative regret over time (ndarray): cum sum of the regret
    '''
    # tools for the arms
    n_arms = len(mab)
    arm_means = np.zeros(n_arms)
    for i, arm in enumerate(mab):
        arm_means[i] = arm.mean()
    best_means_sum = sum(arm_means[np.argsort(arm_means)[::-1][:n_users]])

    # Initialisating stocked variables
    # sample-mean availability of channel i, as sensed by user j
    sample_mean = np.zeros((n_arms, n_users))

    # number of times user j draws arm i
    drawn_arms = np.ones((n_arms, n_users))

    # cumulative reward obtained per player j using arm i
    reward_per_player = np.zeros((n_arms, n_users))

    # bandit algorithm statistic (here UCB)
    decision_statistic = np.zeros((n_arms, n_users))

    # 1 if collision happened for player j with arm i
    collision_matrix = np.zeros((n_arms, n_users))

    # which ranked arm is currently selected by player j.
    # Starts with an arangement to avoid collisions
    current_arm_selection = np.zeros(n_users)

    # which ranked arm will be selected by player j next round.
    next_arm_to_play = np.zeros(n_users, dtype='int32')

    # ranking of the favorites arms per player.
    # if favorite_arm(i, j) = x, x is the i^th favorite arm for player j
    favorite_arm = np.zeros((n_users, n_users), dtype='int32')

    # index of arm to play for player j
    index_arm_to_play = np.zeros(n_users, dtype='int32')

    for i in range(n_users):
        index_arm_to_play[i] = i

    # regret accumulated
    regret = np.zeros(t_horizon+1)

    # Drawing all arms ones per user
    for user in range(n_users):
        for i in range(n_arms):
            sample_mean[i, user] = mab[i].draw()
        decision_statistic[:, user] = sample_mean[:, user]/drawn_arms[:, user]
        favorite_arm[:, user] = np.argsort(
                                           decision_statistic[:, user]
                                          )[::-1][:n_users]

    # Loop until t_horizon is achieved
    for t in range(t_horizon):
        cumulative_reward_per_slot = 0
        for user in range(n_users):
            # player 'user' draws arm 'index_arm_to_play[user]'
            drawn_arms[index_arm_to_play[user], user] += 1

            # if there is a collision draw a random arm for the next round
#            print "user:", user
#            print "index_arm_to_play:", index_arm_to_play
#            print "next_arm_to_play", next_arm_to_play
#            print "np.count", np.count_nonzero(index_arm_to_play == index_arm_to_play[user])
            if np.count_nonzero(
                                index_arm_to_play ==
                                index_arm_to_play[user]) != 1:
                collision_matrix[index_arm_to_play[user], user] = 1
                current_arm_selection[user] = random.randrange(n_users)
                print current_arm_selection[user]

            # else receive the associate reward
            else:
                print "we are drawing arm", index_arm_to_play[user]
                reward_drawn = mab[
                                    index_arm_to_play[user]
                                  ].draw()
                cumulative_reward_per_slot += reward_drawn
                reward_per_player[
                                  index_arm_to_play[user],
                                  user
                                  ] += reward_drawn
                sample_mean[
                            index_arm_to_play[user],
                            user
                            ] = reward_per_player[
                                                  index_arm_to_play[user],
                                                  user
                                                  ]/drawn_arms[
                                                  index_arm_to_play[user],
                                                  user
                                                  ]
                decision_statistic[:, user] = sample_mean[:, user] + \
                    np.sqrt(2*np.log(t+1) / drawn_arms[:, user])
                print decision_statistic[:, user]
                favorite_arm[:, user] = np.argsort(
                                            decision_statistic[:, user]
                                                  )[::-1][:n_users]
                print favorite_arm[:, user]
            # Update arm to draw next round
            next_arm_to_play[user] = favorite_arm[
                                                  current_arm_selection[user],
                                                  user
                                                  ]
            print next_arm_to_play[user]
        index_arm_to_play = next_arm_to_play.copy()
        regret[t+1] = regret[t] + best_means_sum - \
            cumulative_reward_per_slot
        regret[t+1]
#        import pdb; pdb.set_trace()
    return regret

# ploting the regret
regret = rho_rand_strategy(mab, n_users, t_horizon)
plt.plot(range(t_horizon+1), regret, linewidth=2)
plt.legend("regret")
plt.show()
