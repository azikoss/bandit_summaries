import numpy as np


def UpperConfidenceBound(bandit, n, delta):
    means = np.array([0] * bandit.K(), dtype=float)

    # pulls each arm once
    for t in range(bandit.K()):
        means[t] += bandit.pull(t)
    total_pulls = np.array([1] * bandit.K(), dtype=float)

    for t in range(bandit.K(), n):
        ucbs = means + calculate_ucb(total_pulls, delta)
        arm = np.random.choice(np.argwhere(ucbs == np.max(ucbs)).flatten())
        means[arm] = ((means[arm] * total_pulls[arm]) + bandit.pull(arm)) / (
            total_pulls[arm] + 1
        )
        total_pulls[arm] += 1


def calculate_ucb(total_pulls, delta):
    return np.sqrt(2 * np.log(1 / delta) / total_pulls)


from bandit.bernoulli import BernoulliBandit

UpperConfidenceBound(BernoulliBandit(means=[0.5, 0.4]), 1000, 1 / 1000 ** 2)
