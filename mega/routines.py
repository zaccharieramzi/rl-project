import numpy as np
import random

from arms import ArmBernoulli
from .users import SecondaryUser, UCBUser, TSUser


def mega_routine(n_users, params, n_arms, t_horizon, arm_means, alg='ucb'):

    arms = [ArmBernoulli(mean) for mean in arm_means]

    algMap = {
        "ucb": UCBUser,
        "ts": TSUser,
        "eps": SecondaryUser
    }
    if alg not in algMap:
        print("alg must be one of", algMap.keys())
    users = [algMap[alg](n_arms, params) for i in range(n_users)]
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
                    user.persistence_proba = user.params[
                        'persistence_proba_init']
                    # mark arm as unavailable
                    user.available_arms[user.arm] = t + \
                        t**user.params['beta'] * random.random()
                    # check if conflict is resolved
                    user.collided = False
                    occupation[i] = -1
                    if (occupation == user.arm).sum() == 1:
                        users[np.where(occupation == user.arm)[0]].collided = \
                            False

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
    return rewards, collisions
