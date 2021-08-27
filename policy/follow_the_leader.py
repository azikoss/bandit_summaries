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
        ix_pull = np.random.choice(np.argwhere(means == np.max(means)).flatten())
        means[ix_pull] = (
            (means[ix_pull] * total_pulls[ix_pull]) + bandit.pull(ix_pull)
        ) / (total_pulls[ix_pull] + 1)
        total_pulls[ix_pull] += 1
