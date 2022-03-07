# 6. Explore-Then-Commit Algorithm
<center>
<img width="480" src="./assets/6_explore_then_commit.png">
</center>

Explore-then-commit (ETC) is a simple algorithm that **explores each arm a fixed number of times and then commits to playing the arm that performed the best during exploration**. ETC is determined by the number of times <img src="https://render.githubusercontent.com/render/math?math=m"> that each arm is explored. Since there are <img src="https://render.githubusercontent.com/render/math?math=k"> arms and each arm is explored <img src="https://render.githubusercontent.com/render/math?math=m"> times, the algorithm explores <img src="https://render.githubusercontent.com/render/math?math=mk"> times in total. 

Let <img src="https://render.githubusercontent.com/render/math?math=\hat{u}_i(t)"> be the average reward received from arm <img src="https://render.githubusercontent.com/render/math?math=i"> after round <img src="https://render.githubusercontent.com/render/math?math=t">. <img src="https://render.githubusercontent.com/render/math?math=\hat{u}_i(t)"> is formally defined as 
<img src="https://render.githubusercontent.com/render/math?math=\hat{u}_i(t) = \frac{1}{T_i(t)}\sum_{s=1}^{t}\mathbb{I}\{A_s = i \}X_s">, where <img src="https://render.githubusercontent.com/render/math?math=T_i(t) = \sum_{s=1}^{t}\mathbb{I}\{A_s=i\}"> expresses the number of times arm <img src="https://render.githubusercontent.com/render/math?math=i"> has been played after round <img src="https://render.githubusercontent.com/render/math?math=t">. The ETC policy goes as follows
1. Input: <img src="https://render.githubusercontent.com/render/math?math=m">
2. In round <img src="https://render.githubusercontent.com/render/math?math=t"> choose arm <img src="https://render.githubusercontent.com/render/math?math=A_t"> as <img src="https://render.githubusercontent.com/render/math?math=(t \mod k) \%2B 1"> if <img src="https://render.githubusercontent.com/render/math?math=t \leq mk"> otherwise as <img src="https://render.githubusercontent.com/render/math?math=\argmax_i\hat{\mu_i}(mk)">.

We implemented the ETC algorithm in the similar format as the [implementation](4_stochastic_bandits.html#unstructured-bandits))of the Follow-The-Leader algorithm.

```python
def ExploreThenCommit(bandit, n, m):
    means = np.array([0] * bandit.K())

    # explore
    for t in range(bandit.K() * m):
        arm = t % bandit.K()
        means[arm] += bandit.pull(arm)
    means = means / m
    
    # commit 
    for t in range(bandit.K() * m, n):        
        bandit.pull(
            np.random.choice(np.argwhere(means == np.max(means)).flatten())
        )
``` 

## Regret
Recall that <img src="https://render.githubusercontent.com/render/math?math=u_i"> is the true mean reward of action <img src="https://render.githubusercontent.com/render/math?math=i"> and <img src="https://render.githubusercontent.com/render/math?math=\Delta_i = \mu* - \mu_i"> is the suboptimality gap between the optimal arm and arm <img src="https://render.githubusercontent.com/render/math?math=i">.

>When ETC is interacting with any 1-subgaussion bandit and <img src="https://render.githubusercontent.com/render/math?math=1 \leq m \leq n/k">, <img src="https://render.githubusercontent.com/render/math?math=R_n \leq m \sum_{i=1}^{k}\Delta_i \%2B (n - mk)\sum_{i=1}^{k}\Delta_i \exp(-\frac{m\Delta_i^2}{4})"> 


The proof of the above theorem goes as follows
1. By the regret [decomposition lemma](4_stochastic_bandits.md#decomposing-the-regret), regret of any bandit algorithm can be written as <img src="https://render.githubusercontent.com/render/math?math=\sum_{i=1}^{k} \Delta_i \mathbb{E}[T_i(n)]">.
1. For ETC, <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[T_i(n)] = m \%2B (n-mk)\mathbb{P}(A_{mk %2B 1} = i)"> which reflects that each arm is played <img src="https://render.githubusercontent.com/render/math?math=m"> times during the exploration and in the remaining <img src="https://render.githubusercontent.com/render/math?math=n - mk"> exploitation rounds each arm is played with a certain probability
1. <img src="https://render.githubusercontent.com/render/math?math=\mathbb{P}(A_{mk %2B 1} = i) \leq \mathbb{P}(\hat{\mu}_i(mk) \geq \max_{j \neq i} \hat{\mu_j}(mk))"> // the probability of playing arm <img src="https://render.githubusercontent.com/render/math?math=i"> during exploitation can be bounded by the probability that arm <img src="https://render.githubusercontent.com/render/math?math=i"> after the last exploration round has the same (or higher) sample reward mean than the arm with the maximum sample reward. The probability on the right hand side is the same or higher than the probability on the left hand side because multiple arms at time <img src="https://render.githubusercontent.com/render/math?math=mk"> can have the highest sample reward mean.  
1. <img src="https://render.githubusercontent.com/render/math?math=\leq \mathbb{P}(\hat{\mu}_i(mk) \geq \hat{\mu_1}(mk))"> // arm 1 was assumed to be the optimal so that <img src="https://render.githubusercontent.com/render/math?math=\mu_1=\mu*=\max_i\mu_i">
1. <img src="https://render.githubusercontent.com/render/math?math== \mathbb{P}(\hat{\mu}_i(mk) - \mu_i - (\hat{\mu_1}(mk) - \mu_1) \geq \Delta_i)"> // suboptimality gap <img src="https://render.githubusercontent.com/render/math?math=\Delta_i"> was added to the right hand side within the probability and <img src="https://render.githubusercontent.com/render/math?math=\mu_1 - \mu_i">, which equals to <img src="https://render.githubusercontent.com/render/math?math=\Delta_i">, to the left hand side.
1. <img src="https://render.githubusercontent.com/render/math?math=\leq \exp(-\frac{m\Delta_{i}^2}{4})"> // the theorem <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[e^{\lambda X}] \leq e^{\lambda^2 \sigma^2 / 2}"> that bounds subgaussions was used given that <img src="https://render.githubusercontent.com/render/math?math=\hat{\mu}_i(mk) - \mu_i - (\hat{\mu_1}(mk) - \mu_1)"> is <img src="https://render.githubusercontent.com/render/math?math=\sqrt{2/m}">-subgaussian because (following this [proof](5_concentration_of_measure.md#bounding-the-sample-reward-mean)) <img src="https://render.githubusercontent.com/render/math?math=\hat{\mu}_i(mk) - \mu_i"> and <img src="https://render.githubusercontent.com/render/math?math=\hat{\mu_1}(mk) - \mu_1"> are both <img src="https://render.githubusercontent.com/render/math?math=1/\sqrt{m}">-subgaussians and their sum is <img src="https://render.githubusercontent.com/render/math?math=\sqrt{2/m}">-subgaussian  

Substituting row 6. into row 2. and then the result to row 1. finalizes the proof.  

Let's note that the above inequality can be adjusted  when the reward distribution of all arms is not 1-subgaussion. 1-subgaussion is used for the sake of not writing the subgaussion constant <img src="https://render.githubusercontent.com/render/math?math=\sigma"> over. The subgaussion parameter <img src="https://render.githubusercontent.com/render/math?math=\sigma"> however has to be known by the learner for the above regret bound to hold.


## Exploration length
The bound of the ETC algorithm derived above illustrates the trade-off between exploration and exploitation. If  <img src="https://render.githubusercontent.com/render/math?math=m"> is large, the algorithm would explore for too long, and the first term would be too large. If <img src="https://render.githubusercontent.com/render/math?math=m"> is too small, then the probability that the algorithm commits to exploiting the wrong arm is high, and the second term becomes large. So how to choose <img src="https://render.githubusercontent.com/render/math?math=m">?

Let's illustrate it on an example where <img src="https://render.githubusercontent.com/render/math?math=k=2"> and where the first arm is optimal so <img src="https://render.githubusercontent.com/render/math?math=\Delta_1 = 0"> and <img src="https://render.githubusercontent.com/render/math?math=\Delta = \Delta_2">. Then, the bound of the ETC  simplifies to <img src="https://render.githubusercontent.com/render/math?math=R_n \leq m\Delta_i \%2B (n - 2m)\Delta \exp(-\frac{m\Delta_i^2}{4}) \leq m\Delta_i \%2B n\Delta \exp(-\frac{m\Delta_i^2}{4})">. The right hand side expression was obtained by removing <img src="https://render.githubusercontent.com/render/math?math=-2m\Delta \exp(-\frac{m\Delta_i^2}{4})"> from the left hand side expression. For large <img src="https://render.githubusercontent.com/render/math?math=n"> the right hand side expression can be minimized by <img src="https://render.githubusercontent.com/render/math?math=m=\max\{1, \frac{4}{\Delta^2}\log(\frac{n\Delta^2}{4})\}">. This would lead to the upper regret bound of  <img src="https://render.githubusercontent.com/render/math?math=O(\sqrt{n})">. I did not fully understand the intermediate calculations. They are included in the book and also at this [blog post](https://banditalgs.com/2016/09/14/first-steps-explore-then-commit/#mjx-eqn-eqregret_g). In any case, the above derived regret bound of  <img src="https://render.githubusercontent.com/render/math?math=O(\sqrt{n})"> relies on the knowledge of the suboptimality gap <img src="https://render.githubusercontent.com/render/math?math=\Delta"> and horizon <img src="https://render.githubusercontent.com/render/math?math=n">. While the horizon can be known beforehand (and if not the doubling trick can be applied), suboptimally gaps are not. The regret bound of the ETC is <img src="https://render.githubusercontent.com/render/math?math=O(n^{2/3})"> when not relying on the knowledge of the suboptimality gaps. Such a bound is **gap/problem/distribution/instance dependent** since it only depends on the knowledge of the horizon and bandit class, and not on the specific instance within the bandit class.

We let the ETC algorithm play a Bernoulli bandit with <img src="https://render.githubusercontent.com/render/math?math=k=2"> arms and reward means <img src="https://render.githubusercontent.com/render/math?math=\mu_1=0.5"> and <img src="https://render.githubusercontent.com/render/math?math=\mu_2=\mu_1-\Delta"> where <img src="https://render.githubusercontent.com/render/math?math=\Delta"> is sampled from the interval of <img src="https://render.githubusercontent.com/render/math?math=[0, 0.5]">. The horizon was <img src="https://render.githubusercontent.com/render/math?math=n=5000">. We set the exploration length <img src="https://render.githubusercontent.com/render/math?math=m"> optimally by the formula above as well as with several arbitrarily chosen exploration lengths of 5, 25, and 125. The figure below shows the expected reward for the ETC with varying exploration length. Each point in the figure is a mean of 250 simulations. 

<figure class="image" align="center">
  <img src="./assets/6_etc_regrets.png" alt="Regret of the follow-the-leader policy">
</figure>  

 

## Future topics
The book includes exercises covering topics that I would like to understand, particularly 
* Ex. 6.5.: choosing the exploration length <img src="https://render.githubusercontent.com/render/math?math=m"> only based on the knowledge of the horizon length
* Ex. 6.6.: converting an algorithm whose upper bound is dependent on the knowledge of the horizon into an algorithm that is horizon free

I plan to come back to the above exercises. Please share your solutions if resolve them before me. If you have any questions or comments, I would be happy if you write them in the [discussion](https://github.com/azikoss/bandit_summaries/discussions/categories/6-explore-then-commit) section. 



 
# References
This text is *my* summary from the 6. Chapter of [Bandit Algorithm](https://tor-lattimore.com/downloads/book/book.pdf) book. The summary may contain copy&pasted text from the book. 
