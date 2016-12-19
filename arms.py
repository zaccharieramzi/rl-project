import random
import itertools
import numpy as np


class ArmBernoulli:
    ''' Arm to be played in the bandit model with Bernoulli Distribution
    '''
    def __init__(self, p):
        '''Args:
            - p (float): the paramater of the Bernoulli distribution
        '''
        if 0 <= p <= 1:
            self.p = p
        else:
            raise ValueError("The parameter of a Bernoulli distribution must be\
                between 0 and 1")

    def draw(self):
        if random.random() < self.p:
            return 1
        else:
            return 0

    def mean(self):
        return self.p


class ArmUniform:
    ''' Arm to be played in the bandit model with uniform Distribution
    '''
    def draw(self):
        return random.random()

    def mean(self):
        return 0.5


class ArmMultinomial:
    ''' Arm to be played in the bandit model with multinomial Distribution
    '''
    def __init__(self, points, probabilities):
        ''' Args:
            - points (list of float): the points which have a probability > 0
            - probabilities (list of float): the probabilities of each point in
                the mutlinomial distribution
        '''
        if all((0 <= x <= 1 for x in points)):
            self.points = np.array(points)
        else:
            raise ValueError("The points of the multinomial have to be bounded\
                between 0 and 1")
        if all((0 <= p <= 1 for p in probabilities)):
            if sum(probabilities) == 1:
                self.probabilities = np.array(probabilities)
            else:
                raise ValueError("The probabilities of the multinomial have to\
                    sum to 1")
        else:
            raise ValueError("The probabilities of the multinomial have to be bounded\
                between 0 and 1")

    def draw(self):
        cumulative_probabilities = np.cumsum(self.probabilities)
        p = random.random()
        # We try to get the first index where the cummulative probability is
        # greater than p
        i = next(idx for (idx, cumul_proba)
                 in enumerate(cumulative_probabilities)
                 if cumul_proba > p)
        return self.points[i]

    def mean(self):
        return np.sum(self.probabilities * self.points)


class ArmBeta:
    ''' Arm to be played in the bandit model with beta Distribution
    '''
    def __init__(self, alpha, beta):
        if (alpha > 0) and (beta > 0):
            self.alpha = alpha
            self.beta = beta
        else:
            raise ValueError("The paramaters of the beta distribution have to \
                be positive")

    def draw(self):
        return random.betavariate(self.alpha, self.beta)

    def mean(self):
        return self.alpha / (self.alpha + self.beta)
