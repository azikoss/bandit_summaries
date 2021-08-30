# 4. Stochastic Bandits
<img width="" src="./assets/4_dices_small.jpg">

This chapter revisits the introduced definitions about regret, learning objectives, environment class in the context of stochastic bandits. The introduced theory is a basis for the next chapters about stochastic bandits.  

A **stochastic bandit** <img src="https://render.githubusercontent.com/render/math?math=v"> is a collection of distributions <img src="https://render.githubusercontent.com/render/math?math=(P_a: a \in A)"> where <img src="https://render.githubusercontent.com/render/math?math=A"> is the set of available actions. The learner and the environment interact sequentially over <img src="https://render.githubusercontent.com/render/math?math=n"> rounds. In each round <img src="https://render.githubusercontent.com/render/math?math=t \in \{1,2,...,n\}">, the learner chooses an action <img src="https://render.githubusercontent.com/render/math?math=A_t \in \A">. The environment then samples reward <img src="https://render.githubusercontent.com/render/math?math=X_t \in \mathbb{R}"> from the distribution <img src="https://render.githubusercontent.com/render/math?math=P_{A_t}">. The learner cannot see the future observations when making current decisions. 

As mentioned in the [Introduction chapter](1_introduction.md), the learner's objective is to choose actions that lead to the largest possible cumulative reward over all <img src="https://render.githubusercontent.com/render/math?math=n"> rounds. This task is **not an optimization problem** mainly because the learner does not know the distribution for each arm, in other words the bandit instance <img src="https://render.githubusercontent.com/render/math?math=v = (P_a: a \in A)"> is unknown to the learner. Other reason why a bandit problem is not an optimization problem is that the value of <img src="https://render.githubusercontent.com/render/math?math=n"> is not known. This could be however overcome by designing a policy with a fixed horizon and then adapting it for the unknown horizon while proving that the performance loss of this operation is minimal.  
 
## Types of Environment Classes
The book distinguishes between **structured** and **unstructured** environment classes or in other words structured and unstructured bandits.  

### Unstructured Bandits
An environment class <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> is unstructured if the **set of arms <img src="https://render.githubusercontent.com/render/math?math=A"> is finite and there exists sets of distributions <img src="https://render.githubusercontent.com/render/math?math=M_a"> for each <img src="https://render.githubusercontent.com/render/math?math=a \in A"> such that** <img src="https://render.githubusercontent.com/render/math?math=\varepsilon = \{v = (P_a: a \in A): P_a \in M_a"> for all <img src="https://render.githubusercontent.com/render/math?math=\a \in \A\}">. You can read the above formula such that each arm <img src="https://render.githubusercontent.com/render/math?math=a"> has its own function <img src="https://render.githubusercontent.com/render/math?math=P_a"> for distribution of reward. By playing action <img src="https://render.githubusercontent.com/render/math?math=a">, the learner thus cannot learn anything about other actions <img src="https://render.githubusercontent.com/render/math?math=b \neq a">.

Typical unstructured stochastic bandit is a Bernoulli bandit <img src="https://render.githubusercontent.com/render/math?math=\varepsilon_{B}^k = \{(B(\mu_i))_i : \mu \in [0,1]^k \}">. The implementation of the Bernoulli bandit (Exercise 4.7 from the book) is in the code block below ([github link](https://github.com/azikoss/bandit_summaries/blob/main/bandit/bernoulli.py)). Another classic example is a Gaussian bandit (with unknown variance) <img src="https://render.githubusercontent.com/render/math?math=\varepsilon_{N}^k = \{(N(\mu_i, \sigma_{i}^2))_i : \mu \in \mathbb{R}^k "> and <img src="https://render.githubusercontent.com/render/math?math=\sigma^2 \in [0,\inf)^k \}">. These two examples are **parametric** environment classes because the number of degrees of freedom that defines them is finite, otherwise they would be **non-parametric**.


```python
class BernoulliBandit:
    def __init__(self, means):
        """Accepts a list of K >= 2 floats , each lying in [0 ,1]"""
        self._means = means
        self._max_mean = np.max(self._means)
        self._acc_pseudo_regret = 0

    def K(self):
        """Returns the number of arms"""
        return len(self._means)

    def pull(self, a):
        """Accepts a parameter 0 <= a <= K -1 and returns the
        realisation of random variable X with P(X = 1) being
        the mean of the (a + 1) th arm ."""
        reward = np.random.binomial(1, self._means[a])
        self._acc_pseudo_regret += self._max_mean - self._means[a]
        return reward

    def regret(self):
        """Returns the (pseudo) regret incurred so far"""
        return self._acc_pseudo_regret
```

The knowledge (or the assumption that a learner makes) about an environment class influences the performance. With a larger environmental class, it is more difficult to achieve good performance.

### Structured Bandits
**Environment classes that are not unstructured are structured**. In a structured environment class, a learner can obtain information about some actions while never playing them. 

A simple example of an unstructured environment class with <img src="https://render.githubusercontent.com/render/math?math=A = \{1,2\}"> is <img src="https://render.githubusercontent.com/render/math?math=\varepsilon = \{(\mathrm{B}(\theta)), \mathrm{B}(1-\theta): \theta \in [0,1] \}">. In this environment class whose prescription for <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> is known to the learner (but not the parameter of <img src="https://render.githubusercontent.com/render/math?math=\theta">), the learner can  learn the mean of both arms by playing just one arm. Another common unstructured bandits are stochastic linear bandits. 

## The Regret
In the [Introduction chapter](1_introduction.md), the regret was informally defined as the deficit suffered by the learner relative to the optimal policy. Let's revisit this definition. Let <img src="https://render.githubusercontent.com/render/math?math=v = (P_a: a \in A)"> be a stochastic bandit and let define the mean reward of arm <img src="https://render.githubusercontent.com/render/math?math=a"> as <img src="https://render.githubusercontent.com/render/math?math=\mu_{a}(v) = \int_{\infinity}^{-\infinity} x \mathrm{d} P_a(x)">. Then, let <img src="https://render.githubusercontent.com/render/math?math=\mu^*(v) = \displaystyle\max_{a \in A} \mu_a(v)"> be the largest mean of all the arms. Note that <img src="https://render.githubusercontent.com/render/math?math=\mu_a(v)"> is formally defined as a function of the bandit instance <img src="https://render.githubusercontent.com/render/math?math=v">. When the context is clear, <img src="https://render.githubusercontent.com/render/math?math=v"> is omitted from the definition. 

**The regret of policy <img src="https://render.githubusercontent.com/render/math?math=\pi"> on bandit instance <img src="https://render.githubusercontent.com/render/math?math=v"> is <img src="https://render.githubusercontent.com/render/math?math=R_n(\pi, v) = n\mu^*(v) - \mathbf{E}[\sum_{t=1}^{\n} X_t]">**, where the expectation is taken with respect to the probability outcomes induced by the interaction of <img src="https://render.githubusercontent.com/render/math?math=\pi"> and <img src="https://render.githubusercontent.com/render/math?math=v">.

In stochastic bandit environment
 - the regret is always **non-negative**
 - **exists** a policy that has **zero regret** (optimal policy)
 - achieving **zero regret** is possible if an only if the learner **knows the optimal arm** upfront 
 
 Regarding the last point, a relatively weak objective is to find a policy <img src="https://render.githubusercontent.com/render/math?math=\pi"> with sublinear regret on all <img src="https://render.githubusercontent.com/render/math?math=v \in \varepsilon">. Formally, this objective is to find a policy <img src="https://render.githubusercontent.com/render/math?math=\pi"> for which <img src="https://render.githubusercontent.com/render/math?math=$\lim_{n \to \infinity} \dfrac{R_n(\pi, v)}{n} = n"> for all <img src="https://render.githubusercontent.com/render/math?math=v \in \varepsilon">. In such case, the learner is choosing the optimal action almost all of the time as the horizon goes to infinity. 

An alternative how to define regret is to decompose <img src="https://render.githubusercontent.com/render/math?math=R_n"> into a function of the bandit instance <img src="https://render.githubusercontent.com/render/math?math=C: \varepsilon \to [0, \infinity]"> and a function of the horizon <img src="https://render.githubusercontent.com/render/math?math=f: \mathbb{N} \to [0, \infinity)"> such that for all <img src="https://render.githubusercontent.com/render/math?math=n \in \mathbb{N}, v \in \varepsilon">, <img src="https://render.githubusercontent.com/render/math?math=R_n(\pi, v) \leq C(v)f(n)">. 

### Decomposing the Regret
This section presents a lemma about regret decomposition that forms a basis for majority of proofs for stochastic bandits.  

Let <img src="https://render.githubusercontent.com/render/math?math=v = (P_a: a \in A)"> be a stochastic bandit and define **suboptimality gap** or **action gap** or **immediate regret** of action <img src="https://render.githubusercontent.com/render/math?math=a">  as <img src="https://render.githubusercontent.com/render/math?math=\Delta_a(v) = u^*(v) - u_a(v)">. Further, let <img src="https://render.githubusercontent.com/render/math?math=T_a(t) = \sum_{s=1}^{\t} \mathbb{1} \{A_s = a\}"> be the number of times action <img src="https://render.githubusercontent.com/render/math?math=a"> was chosen by the learner after the end of the round <img src="https://render.githubusercontent.com/render/math?math=t">. <img src="https://render.githubusercontent.com/render/math?math=T_a(t)"> is random even with a deterministic policy that chooses the same action for a given history. This is because <img src="https://render.githubusercontent.com/render/math?math=A_s">  depends on the rewards observed in the previous rounds, which are random, so <img src="https://render.githubusercontent.com/render/math?math=A_s"> and consequently <img src="https://render.githubusercontent.com/render/math?math=T_a(t)"> inherit the randomness.

>The regret decomposition lemma states that for any policy <img src="https://render.githubusercontent.com/render/math?math=\pi"> and stochastic bandit <img src="https://render.githubusercontent.com/render/math?math=v"> with finite <img src="https://render.githubusercontent.com/render/math?math=A"> and horizon <img src="https://render.githubusercontent.com/render/math?math=n \in \mathbb{N}">, the regret <img src="https://render.githubusercontent.com/render/math?math=R_n"> of policy <img src="https://render.githubusercontent.com/render/math?math=\pi"> in <img src="https://render.githubusercontent.com/render/math?math=v"> satisfies 
<img src="https://render.githubusercontent.com/render/math?math=R_n = \sum_{a \in A} \Delta_a \mathbb{E}[T_a(n)]">.

The lemma decomposes the regret with respect to the losses of each arm. This tells us that to keep the regret small, the learner should try to use an arm with a large suboptimally gap proportionally fewer times.

#### Proof
We will prove that the regret defined by summing over time steps <img src="https://render.githubusercontent.com/render/math?math=R_n = n\mu^* - \mathbb{E}[\sum_{t=1}^{\n} X_t]"> is equivalent to the definition <img src="https://render.githubusercontent.com/render/math?math=\sum_{a \in A} \Delta_a \mathbb{E}[T_a(n)]"> from the lemma. The proof goes as follows

1. <img src="https://render.githubusercontent.com/render/math?math=R_n = n\mu^* - \mathbb{E}[\sum_{t=1}^{\n} X_t]"> 
1. <img src="https://render.githubusercontent.com/render/math?math== \color{green}\sum_{t=1}^{n}\mathbb{E}[(u^* - X_t)]"> // the summation was moved outside of the expected value and <img src="https://render.githubusercontent.com/render/math?math=\mu^*"> was moved inside the expected value 
1. <img src="https://render.githubusercontent.com/render/math?math== \color{green}\sum_{a \in A} \color{black}\sum_{t=1}^{n}\mathbb{E}[(u^* - X_t)\color{green}\mathbb{I}\{A_t = a\}\color{black}]"> // an indicator function <img src="https://render.githubusercontent.com/render/math?math=\mathbb{I}"> was added to the formula without changing its value since  <img src="https://render.githubusercontent.com/render/math?math=\sum_{a \in A}\mathbb{I}\{A_t = a\} = 1"> at time <img src="https://render.githubusercontent.com/render/math?math=t"> 
1. <img src="https://render.githubusercontent.com/render/math?math== \sum_{a \in A} \sum_{t=1}^{n}\mathbb{E}[(u^* - X_t)\mathbb{I}\{A_t = a\}|\color{green}A_t)\color{black}] \color{green} P(A_t=a)"> // the expectation was conditioned by <img src="https://render.githubusercontent.com/render/math?math=A_t">
1. <img src="https://render.githubusercontent.com/render/math?math== \sum_{a \in A} \sum_{t=1}^{n}\color{green}\mathbb{I}\{A_t = a\}\color{black}\mathbb{E}[(u^* - X_t)|A_t] P(A_t=a)"> // the indicator function was taken out of the expectation since action <img src="https://render.githubusercontent.com/render/math?math=A_t"> is given
1. <img src="https://render.githubusercontent.com/render/math?math== \sum_{a \in A} \sum_{t=1}^{n}\mathbb{I}\{A_t = a\}\color{green}(u^* - u_{A_t})\color{black} P(A_t=a)"> // the expectation was removed since <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[X_t|A_t] = u_{A_t}">
1. <img src="https://render.githubusercontent.com/render/math?math== \sum_{a \in A} \sum_{t=1}^{n}\mathbb{I}\{A_t = a\}(u^* - \color{green}u_{A_a}\color{black}) P(A_t=a)"> // <img src="https://render.githubusercontent.com/render/math?math=u_{A_t}"> was written as <img src="https://render.githubusercontent.com/render/math?math=u_{a}"> since the indication function zeros out all actions apart from action <img src="https://render.githubusercontent.com/render/math?math=a">
1. <img src="https://render.githubusercontent.com/render/math?math== \sum_{a \in A} \sum_{t=1}^{n}\mathbb{I}\{A_t = a\}\color{green}\Delta_a\color{black} P(A_t=a)"> // the definition of the action gap was used
1. <img src="https://render.githubusercontent.com/render/math?math== \sum_{a \in A} \color{green}\Delta_a\color{black} \sum_{t=1}^{n}\mathbb{I}\{A_t = a\} P(A_t=a)"> // <img src="https://render.githubusercontent.com/render/math?math=\Delta_a"> was moved outside of the summation over time since it does not depends on time
1. <img src="https://render.githubusercontent.com/render/math?math== \sum_{a \in A} \Delta_a \color{green}T_a(n)\color{black} P(A_t=a)"> // the formula for <img src="https://render.githubusercontent.com/render/math?math=T_a(n)"> was used
1. <img src="https://render.githubusercontent.com/render/math?math== \sum_{a \in A} \Delta_a \color{green}\mathbb{E}[T_a(n)]"> // the formula for expected value was used 
 

### Alternative Definitions
We defined regret as an expectation. If it is desired to measure the variance of the regret caused by randomness, regret can be defined as a **random regret** <img src="https://render.githubusercontent.com/render/math?math=\widetilde{R_n} = n\mu^{*} - \sum_{t=1}^{n}X_t"> or as a **pseudo regret** <img src="https://render.githubusercontent.com/render/math?math=\bar{R_n} = n\mu^{*} - \sum_{t=1}^{n}u_{A_t}">. Since  <img src="https://render.githubusercontent.com/render/math?math=\widetilde{R_n}"> is influenced by the noise <img src="https://render.githubusercontent.com/render/math?math=X_t - u_{A_t}">, **pseudo-regret appears to be a better** performance measure of a bandit policy.
 
## Follow the Leader 
**Follow-the-leader** (or so called greedy algorithm) is an simple policy that chooses **each action once** and then chooses the action with the **largest observed average reward** so far. The goal of the exercise 4.8 from the book is to implement this algorithm. The implementation goes as follows ([github link](https://github.com/azikoss/bandit_summaries/blob/main/policy/follow_the_leader.py)):   

```python
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
```

Running 1000 simulations ([github link](https://github.com/azikoss/bandit_summaries/blob/main/run/4_follow_the_leader_with_bernoulli_bandit.py)) of the follow-the-leader policy on the Bernoulli bandit with two arms and means <img src="https://render.githubusercontent.com/render/math?math=\mu_1=0.5"> and <img src="https://render.githubusercontent.com/render/math?math=\mu_2=0.6"> with the horizon of <img src="https://render.githubusercontent.com/render/math?math=n=100"> steps and plotting the pseudo regret of each simulation yields Figure 1 below (Exercise 4.11). The figure shows that follow-the-leader policy works very well in many cases as the regret is nearly zero but in other cases it does not as the regret is close to the maximum possible pseudo regret of 10 = (0.6-0.5) * 100. Such a volatility in the performance is typical for this policy. The volatility is caused by the limited exploration that often results in the commitment to play a suboptimal arm.  

<figure class="image" align="center">
  <img src="./assets/4_regret.png" alt="Regret of the follow-the-leader policy">
  <figcaption>Figure 1: Regret of the follow-the-leader policy</figcaption>
</figure> 

# References
This text *my* summary from the 4. Chapter of [Bandit Algorithm](https://tor-lattimore.com/downloads/book/book.pdf) book. The summary contains copy&pasted text from the book as well as some additional text. 
