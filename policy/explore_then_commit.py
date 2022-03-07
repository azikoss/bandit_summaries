import numpy as np


def ExploreThenCommit(bandit, n, m):
    means = np.array([0] * bandit.K())

    # explore
    for t in range(bandit.K() * m):
        arm = t % bandit.K()
        means[arm] += bandit.pull(arm)
    means = means / m

    # commit
    for t in range(bandit.K() * m, n):
        bandit.pull(np.random.choice(np.argwhere(means == np.max(means)).flatten()))


def get_optimal_exploration_len_two_arms(n, gap):
    if gap == 0:
        return 1
    return max(1, int(np.ceil(4 / gap ** 2 * np.log((n * gap ** 2) / 4))))
