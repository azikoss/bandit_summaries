import pandas as pd
import matplotlib.pyplot as plt
from bandit.bernoulli import BernoulliBandit
from policy.follow_the_leader import FollowTheLeader

if __name__ == "__main__":
    regrets = []
    for _ in range(1000):
        bandit = BernoulliBandit(means=[0.5, 0.6])
        FollowTheLeader(bandit, n=100)
        regrets.append(bandit.regret())

    # plotting the regret
    ax = pd.Series(regrets).hist()
    ax.set_xlabel("Regret")
    ax.set_ylabel("Frequency")
    plt.show()
