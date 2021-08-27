import numpy as np


def FollowTheLeader(bandit, n):
    means = [0] * bandit.K()

    # pulls each arm once
    for t in range(bandit.K()):
        means[t] += bandit.pull(t)
    total_pulls = [1] * n

    # plays the arm with the highest mean
    for t in range(bandit.K(), n):
        # randomly select one of the arms that has the highest mean
        arm = np.random.choice(np.argwhere(means == np.max(means)).flatten())
        means[arm] = ((means[arm] * total_pulls[arm]) + bandit.pull(arm)) / (
            total_pulls[arm] + 1
        )
        total_pulls[arm] += 1
