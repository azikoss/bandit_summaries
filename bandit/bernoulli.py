import numpy as np


class BernoulliBandit:
    def __init__(self, means):
        """Accepts a list of K >= 2 floats , each lying in [0 ,1]"""
        self._means = means
        self._max_mean = np.max(self._means)
        self._acc_pseudo_regret = 0

    def K(self):
        """Function should return the number of arms"""
        return len(self._means)

    def pull(self, a):
        """Accepts a parameter 0 <= a <= K -1 and returns the
        realisation of random variable X with P(X = 1) being
        the mean of the (a +1) th arm ."""
        reward = np.random.binomial(1, self._means[a])
        self._acc_pseudo_regret += self._max_mean - self._means[a]
        return reward

    def regret(self):
        """Returns the (pseudo) regret incurred so far"""
        return self._acc_pseudo_regret
