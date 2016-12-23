import numpy as np
import random

from arms import ArmBernoulli
from users import SecondaryUser, UCBUser, TSUser

from plots import regret_plt

#  UNIVERSE PARAMETERS
n_users = 3
n_arms = 5
t_horizon = 100000

arm_means = [0.2, 0.3, 0.5, 0.8, 0.9]

arms = list()
for i in range(n_arms):
    arms.append(ArmBernoulli(arm_means[i]))

# ALGORITHM PARAMETERS
params = {
    'c': 0.1,
    'd': 0.05,
    'alpha': 0.5,
    'beta': 0.8,
    'persistence_proba_init': 0.6
}

users = list()
for i in range(n_users):
    # users.append(SecondaryUser(n_arms, params))
    users.append(TSUser(n_arms, params))

# global statistics
rewards = np.zeros((n_users, t_horizon))
collisions = np.zeros((n_users, t_horizon), dtype=np.uint8)
occupation = np.zeros(n_users)
# main loop
for t in range(1, t_horizon):
    # update collision status
    for i, user in enumerate(users):
        if not user.collided:
            # increase its persistence probability
            user.persistence_proba *= user.params['alpha']
            user.persistence_proba += user.params['alpha']
        else:
            collisions[i, t-1] = 1
            if random.random() < user.persistence_proba:
                # the users persists !
                user.previous_arm = user.arm
            else:
                # the user drops
                user.persistence_proba = user.params['persistence_proba_init']
                # mark arm as unavailable
                user.available_arms[user.arm] = t + \
                    t**user.params['beta'] * random.random()
                # check if conflict is resolved
                user.collided = False
                occupation[i] = -1
                if (occupation == user.arm).sum() == 1:
                    users[np.where(occupation == user.arm)[0]].collided = False

    # non collided users make a move
    for i, user in enumerate(users):
        if not user.collided:
            # update users whishes
            occupation[i] = user.decision(t)

    # reward and collisions
    for i, user in enumerate(users):
        if (occupation == user.arm).sum() > 1:
            user.collided = True

        else:
            rewards[i, t] = user.draw_from_arm(arms[user.arm])


# regret curve
print(collisions.sum())

chunk = t_horizon // 10
for i in range(10):
    print("{:d} iterations, {:d} collisions".format(
        chunk, collisions[:, i*chunk:(i+1)*chunk].sum()))
regret_plt(arm_means, rewards)
print("average reward {:f}, optimal 2.2".format(rewards.sum(axis=0).mean()))
