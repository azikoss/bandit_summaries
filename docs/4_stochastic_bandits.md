# 4. Stochastic Bandits
A **stochastic bandit** is a collection of distributions <img src="https://render.githubusercontent.com/render/math?math=v = (P_a: a \in A)"> where <img src="https://render.githubusercontent.com/render/math?math=A"> is the set of available actions and <img src="https://render.githubusercontent.com/render/math?math=P_{A_t}"> is the conditional distribution of reward <img src="https://render.githubusercontent.com/render/math?math=X_t"> given <img src="https://render.githubusercontent.com/render/math?math=A_1, X_1, ..., A_{t-1}, X_{t-1}, A_t">. <img src="https://render.githubusercontent.com/render/math?math=A_t \in A"> as introduced earlier is an action chosen at time <img src="https://render.githubusercontent.com/render/math?math=t">.

As mentioned in Chapter 1, the learner's objective is to choose actions that lead to the largest possible cumulative reward over all *n* rounds. This task is not an optimization problem mainly because the learner does not know the distribution for each arm, in other words the bandit instance <img src="https://render.githubusercontent.com/render/math?math=v = (P_a: a \in A)"> is unknown to the learner. A policy that maximizes the cumulative rewards for one bandit instance may perform badly on another one. (As mentioned in Chapter 1) The learner typically has partial information about <img src="https://render.githubusercontent.com/render/math?math=v"> when the learner knows the environment class <img src="https://render.githubusercontent.com/render/math?math=\varepsilon">. 

Other reason why a bandit problem is not an optimization problem is that the values of *n* is not known. This could be however overcome by designing a policy with a fixed horizon and then adapting it for the unknown horizon while proving that the performance loss of this operation is minimal.  
 
# Types of Environment Classes
The book distinguishes between **structured** and **unstructured** environment classes or in other words structured and unstructured bandits.  

## Structured Bandits
An environment class <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> is unstructured if the set of arms <img src="https://render.githubusercontent.com/render/math?math=A"> is finite and there exists sets of distributions <img src="https://render.githubusercontent.com/render/math?math=M_a"> for each <img src="https://render.githubusercontent.com/render/math?math=a \in A"> such that

<img src="https://render.githubusercontent.com/render/math?math=\varepsilon = \{v = (P_a: a \in A): P_a \in M_a"> for all <img src="https://render.githubusercontent.com/render/math?math=\a \in \A\}">

You can read the above formula such that each arm <img src="https://render.githubusercontent.com/render/math?math=a"> has its own function for distribution of reward <img src="https://render.githubusercontent.com/render/math?math=P_a">. By playing action <img src="https://render.githubusercontent.com/render/math?math=a">, the learner thus cannot learn anything about other actions <img src="https://render.githubusercontent.com/render/math?math=b \neq a">.

Typical environmental class for stochastic bandits are for instance Bernoulli bandit  <img src="https://render.githubusercontent.com/render/math?math=\varepsilon_{B}^k = \{(B(\mu_i))_i : \mu \in [0,1]^k \}"> or Gaussian bandit (with unknown variance) <img src="https://render.githubusercontent.com/render/math?math=\varepsilon_{N}^k = \{(N(\mu_i, \sigma_{i}^2))_i : \mu \in \mathbb{R}^k "> and <img src="https://render.githubusercontent.com/render/math?math=\sigma^2 \in [0,\inf)^k \}">. These two examples are **parametric** environment classes because the number of degrees of freedom that defines them is finite, otherwise they would be **non-parametric**.

The knowledge (or the assumption that a learner makes) about an environment class influences the performance. With a larger environmental class, it is more difficult to achieve good performance.

## Structured Bandits
Environment classes that are not unstructured are structured. In a structured environment class, a learner can obtain information about some actions while never playing them. 

A simple example of an unstructured environment class with <img src="https://render.githubusercontent.com/render/math?math=A = \{1,2\}"> is <img src="https://render.githubusercontent.com/render/math?math=\varepsilon = \{(\mathrm{B}(\theta)), \mathrm{B}(1-\theta): \theta \in [0,1] \}">. In this environment class whose prescription for <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> is known to the learner but just not the parameter of <img src="https://render.githubusercontent.com/render/math?math=\theta">, the learner can learn the mean of both arms by playing just one arm. 

Another common unstructured bandits are stochastic linear bandits. 


# The Regret
In Chapter 1 regret was informally defined as the deficit suffered by the learner relative to the optimal policy. Let's revisit this definition and add more details. Let <img src="https://render.githubusercontent.com/render/math?math=v = (P_a: a \in A)"> be a stochastic bandit and define the mean reward of arm <img src="https://render.githubusercontent.com/render/math?math=a"> as <img src="https://render.githubusercontent.com/render/math?math=\mu_{a}(v) = \int_{\infinity}^{-\infinity} x \mathrm{d} P_a(x)">. Then, let <img src="https://render.githubusercontent.com/render/math?math=\mu^*(v) = \displaystyle\max_{a \in A} \mu_a(v)"> be the largest mean of all the arms. Note that the mean <img src="https://render.githubusercontent.com/render/math?math=\mu_a(v)"> is formally defined as a function of the bandit instance <img src="https://render.githubusercontent.com/render/math?math=v">. When the context is clear, <img src="https://render.githubusercontent.com/render/math?math=v"> is omitted from the definition.

The regret of policy <img src="https://render.githubusercontent.com/render/math?math=\pi"> on bandit instance <img src="https://render.githubusercontent.com/render/math?math=v"> is <img src="https://render.githubusercontent.com/render/math?math=R_n(\pi, v) = n\mu^*(v) - \mathbf{E}[\sum_{t=1}^{\n} X_t]">, where the expectation is taken with respect to the probability outcomes induced by the interaction of <img src="https://render.githubusercontent.com/render/math?math=\pi"> and <img src="https://render.githubusercontent.com/render/math?math=v">.

In stochastic bandit environment
 - the regret is always non-negative
 - exists a policy that has zero regret (optimal policy)
 - achieving zero regret is possible if an only if the learner knows the optimal arm upfront 
 
 Regarding the last point, a relatively weak objective is to find a policy <img src="https://render.githubusercontent.com/render/math?math=\pi"> with sublinear regret on all <img src="https://render.githubusercontent.com/render/math?math=v \in \varepsilon">. Formally, this objective is to find a policy <img src="https://render.githubusercontent.com/render/math?math=\pi"> for which <img src="https://render.githubusercontent.com/render/math?math=$\lim_{x \to \infinity} \dfrac{R_n(\pi, v)}{n} = n"> for all <img src="https://render.githubusercontent.com/render/math?math=v \in \varepsilon">. In such case, the learner is choosing the optimal action almost all of the time as the horizon goes to infinity. 

An alternative how to define regret is to decompose <img src="https://render.githubusercontent.com/render/math?math=R_n"> into a function of the bandit instance <img src="https://render.githubusercontent.com/render/math?math=C: \varepsilon \to [0, \infinity]"> and a function of the horizon <img src="https://render.githubusercontent.com/render/math?math=f: \mathbb{N} \to [0, \infinity)"> such that for all <img src="https://render.githubusercontent.com/render/math?math=n \in \mathbb{N}, v \in \varepsilon">, <img src="https://render.githubusercontent.com/render/math?math=R_n(\pi, v) \leq C(v)f(n)">. 

*All of the policies are inspired by the simple fact that in order to make the regret small, the policy must diiscover the ....*

## Decomposing the Regret
Now we will present how regret can be decomposed for stochastic bandits. Let <img src="https://render.githubusercontent.com/render/math?math=v = (P_a: a \in A)"> be a stochastic bandit and define **suboptimality gap** or **action gap** or **immediate regret** of action <img src="https://render.githubusercontent.com/render/math?math=a">  as <img src="https://render.githubusercontent.com/render/math?math=\Delta_a(v) = u^*(v) - u_a(v)">. Further, let <img src="https://render.githubusercontent.com/render/math?math=T_a(t) = \sum_{s=1}^{\t} \mathbb{1} \{A_s = a\}"> be the number of times action <img src="https://render.githubusercontent.com/render/math?math=a"> was chosen by the learner after the end of the round <img src="https://render.githubusercontent.com/render/math?math=t">. 

<img src="https://render.githubusercontent.com/render/math?math=T_a(t)"> is random even with a deterministic policy that chooses the same action for a given history. This is because it uses <img src="https://render.githubusercontent.com/render/math?math=A_t"> that depends on the rewards observed in the previous rounds, which are random, so <img src="https://render.githubusercontent.com/render/math?math=A_t"> and consequently <img src="https://render.githubusercontent.com/render/math?math=T_a(t)"> inherits the randomness.

The regret decomposition lemma states that for any policy <img src="https://render.githubusercontent.com/render/math?math=\pi"> and stochastic bandit <img src="https://render.githubusercontent.com/render/math?math=v"> with <img src="https://render.githubusercontent.com/render/math?math=A"> finite and horizon <img src="https://render.githubusercontent.com/render/math?math=n \in \mathbb{N}">, the regret <img src="https://render.githubusercontent.com/render/math?math=R_n"> of policy <img src="https://render.githubusercontent.com/render/math?math=\pi"> in <img src="https://render.githubusercontent.com/render/math?math=v"> satisfies 
<img src="https://render.githubusercontent.com/render/math?math=R_n = \sum_{a \in A} \Delta_a \mathbb{E}[T_a(n)]">.

The lemma decomposes the regret in terms of the loss due to using each of the arms (t. It is useful because it tells us that to keep the regret small, the learner should try to to use an arm with with a large suboptimally gap proportionally fewer times. 

### Proof
We will prove that the regret defined by summing over actions <img src="https://render.githubusercontent.com/render/math?math=R_n = n\mu^* - \mathbb{E}[S_n]"> is equivalent to the definition <img src="https://render.githubusercontent.com/render/math?math=\sum_{a \in A} \Delta_a \mathbb{E}[T_a(n)]"> from above that sums over arms. The proof goes as follows

1. <img src="https://render.githubusercontent.com/render/math?math=R_n = n\mu^* - \mathbb{E}[S_n]">, this is the definition of regret based on the summation over rounds 
1. <img src="https://render.githubusercontent.com/render/math?math== \color{green}\sum_{t=1}^{n}\mathbb{E}[(u^* - X_t)]">, explicitly wrote out the formula for <img src="https://render.githubusercontent.com/render/math?math=S_n"> and moved <img src="https://render.githubusercontent.com/render/math?math=\mu^*"> inside of the summation  
1. <img src="https://render.githubusercontent.com/render/math?math== \color{green}\sum_{a \in A} \color{black}\sum_{t=1}^{n}\mathbb{E}[(u^* - X_t)\color{green}\mathbb{I}\{A_t = a\}\color{black}]">, an indicator function was added to the formula based on the fact that <img src="https://render.githubusercontent.com/render/math?math=\sum_{a \in A}\mathbb{I}\{A_t = a\}] = 1"> which is based on the fact that exactly one action is being played in every round. TODO: better explain
1. <img src="https://render.githubusercontent.com/render/math?math== \sum_{a \in A} \sum_{t=1}^{n}\mathbb{E}[(u^* - X_t)\mathbb{I}\{A_t = a\}|\color{green}A_t)\color{black}] \color{green} * P(A_t=a)">, conditioned the expectation on <img src="https://render.githubusercontent.com/render/math?math=A_t">
1. <img src="https://render.githubusercontent.com/render/math?math== \sum_{a \in A} \sum_{t=1}^{n}\color{green}\mathbb{I}\{A_t = a\}\color{black}\mathbb{E}[(u^* - X_t)|A_t] * P(A_t=a)">, took out the indicator function ouf of the expectation since action <img src="https://render.githubusercontent.com/render/math?math=A_t"> is given.
1. <img src="https://render.githubusercontent.com/render/math?math== \sum_{a \in A} \sum_{t=1}^{n}\mathbb{I}\{A_t = a\}\color{green}(u^* - u_{A_t})\color{black} * P(A_t=a)">, rid of the expectation since <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[X_t|A_t] = u_{A_t}">
1. <img src="https://render.githubusercontent.com/render/math?math== \sum_{a \in A} \sum_{t=1}^{n}\mathbb{I}\{A_t = a\}(u^* - \color{green}u_{A_a}\color{black}) * P(A_t=a)">, rewrote <img src="https://render.githubusercontent.com/render/math?math=u_{A_t}"> into <img src="https://render.githubusercontent.com/render/math?math=u_{a}"> since the indication function will zero out all other actions apart from action <img src="https://render.githubusercontent.com/render/math?math=a">
1. <img src="https://render.githubusercontent.com/render/math?math== \sum_{a \in A} \sum_{t=1}^{n}\mathbb{I}\{A_t = a\}\color{green}\Delta_a\color{black} * P(A_t=a)">, applied the definition of the action gap
1. <img src="https://render.githubusercontent.com/render/math?math== \sum_{a \in A} \color{green}\Delta_a\color{black} \sum_{t=1}^{n}\mathbb{I}\{A_t = a\} * P(A_t=a)">, rearranged the formula since the action gap does not depend on the time
1. <img src="https://render.githubusercontent.com/render/math?math== \sum_{a \in A} \Delta_a \color{green}\mathbb{E}[T_a(n)]">, applied the formula for expected value 

## Alternative Definitions
We defined regret as an expectation. If it is desired to measure the variance of the regret caused by randomness, regret can be defined as a random regret <img src="https://render.githubusercontent.com/render/math?math=\widetilde{R_n} = n\mu^{*} - \sum_{t=1}^{n}X_t"> or as a pseudo regret <img src="https://render.githubusercontent.com/render/math?math=R_n = n\mu^{*} - \sum_{t=1}^{n}u_{A_t}">. Since  <img src="https://render.githubusercontent.com/render/math?math=\widetilde{R_n}"> is influenced by the noise <img src="https://render.githubusercontent.com/render/math?math=X_t - u_{A_t}">, pseudo-regret seems to be a better measure to performance of a bandit policy.
 
# Code
* 4.7. Implement Bernoulli bandit
* 4.8. Implement Follow the Leader algo
* 4.11. and 4.12 run simulations with the bandit 

#Questions
- Why P_A_t is conditional on the history?? 
- check if the notation in the previous chapter is enough or additional needs to be iincludes - perhaps a brief recap would be useful


# References
This text *my* summary from the 4. Chapter of [Bandit Algorithm](https://tor-lattimore.com/downloads/book/book.pdf) book. The summary contains copy&pasted text from the book as well as some additional text. 
