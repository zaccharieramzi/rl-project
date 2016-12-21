import numpy as np
import random

from arms import ArmBernoulli
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

c = 0.1
d = 0.05
alpha = 0.5
beta = 0.8
persistence_proba = 0.6


class SecondaryUser:
    """
    class defining the behaviour and stored statistic of a user
    """

    def __init__(self, n_arms):
        self.persistence_proba = persistence_proba
        # an arm i is available if t >= arms_availability[i]
        self.available_arms = np.zeros(n_arms)
        self.arms_rew = np.zeros(n_arms)
        self.draws = np.zeros(n_arms, dtype='int')

        self.collided = False
        self.previous_arm = -1
        self.arm = -1

    def decision(self, eps, t):
        user.previous_arm = user.arm
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
            # exploit

            # best available empirical average of available arms
            available_arms = self.available_arms < t
            empirical_m = self.arms_rew / np.maximum(self.draws, 1)
            self.arm = np.argmax(available_arms * empirical_m)

        # if an arm has changed reset persistence proba
        if self.arm != self.previous_arm:
            self.persistence_proba = persistence_proba
        return self.arm

users = list()
for i in range(n_users):
    users.append(SecondaryUser(n_arms))

rewards = np.zeros((n_users, t_horizon))
collisions = np.zeros((n_users, t_horizon), dtype=np.uint8)
occupation = np.zeros(n_users)
# main loop
for t in range(1, t_horizon):
    # update parameter for greedy algorithm
    eps = min(1, c * (n_arms ** 2) / (d**2 * (n_arms - 1) * t))

    # update collision status
    for i, user in enumerate(users):
        if not user.collided:
            # increase its persistence probability
            user.persistence_proba *= alpha
            user.persistence_proba += alpha
        else:
            collisions[i, t-1] = 1
            if random.random() < user.persistence_proba:
                # the users persists !
                user.previous_arm = user.arm
            else:
                # the user drops
                user.persistence_proba = persistence_proba
                # mark arm as unavailable
                user.available_arms[user.arm] = t + t**beta * random.random()
                # check if conflict is resolved
                user.collided = False
                occupation[i] = -1
                if (occupation == user.arm).sum() == 1:
                    users[np.where(occupation == user.arm)[0]].collided = False

    # non collided users make a move
    for i, user in enumerate(users):
        if not user.collided:
            # update users whishes
            occupation[i] = user.decision(eps, t)

    # reward and collisions
    for i, user in enumerate(users):
        if (occupation == user.arm).sum() > 1:
            user.collided = True

        else:
            rewards[i, t] = arms[user.arm].draw()
            user.arms_rew[user.arm] += rewards[i, t]
            user.draws[user.arm] += 1

# regret curve
print(collisions.sum())

chunk = t_horizon // 10
for i in range(10):
    print("{:d} iterations, {:d} collisions".format(
        chunk, collisions[:, i*chunk:(i+1)*chunk].sum()))
regret_plt(arm_means, rewards)
print("average reward {:f}, optimal 2.2".format(rewards.sum(axis=0).mean()))
