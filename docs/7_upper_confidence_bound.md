# 7. The Upper Confidence Bound Algorithm
This chapter introduces a simple yet very well functioning algorithm called the upper confidence bound (UCB) algorithm. 
*The algorithm has several advantages over the [explore-then-commit (ETC) algorithm](6_explore_then_commit.md) as it  does not rely on the advance knowledge of the suboptimality gaps and works works well when there are more than two arms. The introduced UCB algorithm depends on the horizon *n* but the version presented in the next chapter does not. 

## The Optimism Principle
The UCB algorithm follows the principle of **optimism in the face of uncertainty**. This principle states that one should act as if the environment is as nice as **plausibly possible**. To give an example, imagine visiting a new country and making a choice whether to try a local restaurant or a well-known multinational chain. You are uncertain about the food of the local restaurant. It could be equally great as equally bad - you do not know because you have never been there. Following the above principle would mean taking a optimistic opinion about the food in the local restaurant and trying it out. Afterwards, you could update your current knowledge about it and make more informed decision next time. 


Following the optimism principle in the context of bandits means estimating the mean reward higher than its true value with high probability. This over-estimate is called upper confidence bound. The intuition why this leas to sublinear regret goes as follows. A suboptimal arm is played only when its upper confidence bound is larger than the upper confidence bound of the optimal arm. But this cannot happen too often because playing a suboptimal arm decrease its upper confidence bound that would eventually fall the one of the optimal arm. 


Let's formalize the above intuition and define the upper confidence bound. Let <img src="https://render.githubusercontent.com/render/math?math=(X_t)_{t=1}^{n}"> be a sequence of independent 1-subgaussion random variables with mean <img src="https://render.githubusercontent.com/render/math?math=\mu"> and <img src="https://render.githubusercontent.com/render/math?math=\hat{\mu}=\frac{1}{n} \Sigma_{t=1}^{n}X_t">. By what we learnt in [Chapter 5](5_concentration_of_measure.md#bounding-the-sample-reward-mean), <img src="https://render.githubusercontent.com/render/math?math=\mathbb{P}(\mu \geq \hat{\mu} %2B \sqrt{\frac{2\log(1/\delta)}{n}}) \leq \delta"> for all <img src="https://render.githubusercontent.com/render/math?math=\delta \in (0,1)">. 

Since the learner makes a decision at time <img src="https://render.githubusercontent.com/render/math?math=t">, defining an upper confidence bound based on the above inequality requires making the terms <img src="https://render.githubusercontent.com/render/math?math=\hat{\mu}"> and <img src="https://render.githubusercontent.com/render/math?math=\n"> to be dependent on <img src="https://render.githubusercontent.com/render/math?math=t">. When making a decision at time step <img src="https://render.githubusercontent.com/render/math?math=t">, the learner has observed <img src="https://render.githubusercontent.com/render/math?math=T_i(t-1)"> samples from arm <img src="https://render.githubusercontent.com/render/math?math=i"> and received rewards from that arm with an empirical mean of <img src="https://render.githubusercontent.com/render/math?math=\hat{\mu_i}(t-1)">. Then a reasonable candidate for "as large as plausibly possible" for the unknown mean of the *i*th arm is <img src="https://render.githubusercontent.com/render/math?math=UCB_i(t-1, \delta)"> = <img src="https://render.githubusercontent.com/render/math?math=\infinity"> if <img src="https://render.githubusercontent.com/render/math?math=T_i(t-1) = 0"> or otherwise <img src="https://render.githubusercontent.com/render/math?math=\hat{\mu}(t-1) + %2B \sqrt{\frac{2\log(1/\delta)}{T_i(t-1)}}">. The expression <img src="https://render.githubusercontent.com/render/math?math=\sqrt{\frac{2\log(1/\delta)}{T_i(t-1)}}"> is called  **confidence width** or **exploration bonus**. 

Now, we can state a version of the UCB algorithm as follows
1. **Input** <img src="https://render.githubusercontent.com/render/math?math=k"> and <img src="https://render.githubusercontent.com/render/math?math=\delta">
1. **for** <img src="https://render.githubusercontent.com/render/math?math=t"> and <img src="https://render.githubusercontent.com/render/math?math=t \in 1, ..., n"> **do**
1. &emsp; Choose action <img src="https://render.githubusercontent.com/render/math?math=A_t = argmax_i UCB_i(t-1, \delta)">
1. &emsp; Observe reward <img src="https://render.githubusercontent.com/render/math?math=X_t"> and update upper confdience bounds
1. **end for**

The above algorithm is an **index algorithm**. An index algorithm chooses the arm in each round that maximizes some value, called the **index**. For the UCB algorithm, the index of arm <img src="https://render.githubusercontent.com/render/math?math=i"> is <img src="https://render.githubusercontent.com/render/math?math=UCB_i(t-1, \delta)">. 

<img src="https://render.githubusercontent.com/render/math?math=\delta"> is a called the **confidence level** and it quantifies the degree of certainty. <img src="https://render.githubusercontent.com/render/math?math=\delta"> should be small enough to ensure optimism with high probab1ility but not so large that the suboptimal arms are explored too frequently. Choosing the confidence level will be done in future chapters. For now, the choice of this parameter is done based on the following considerations. If the confidence interval fails and the index of an optimal arm drops belows its true mean, then it could happen that the algorithm stops playing the optimal arm and suffers linear regret. This suggest choosing <img src="https://render.githubusercontent.com/render/math?math=\delta \approx 1/n"> so that playing during a larger horizon would mean less chance of suffer from this failure since the smaller value of <img src="https://render.githubusercontent.com/render/math?math=\delta"> leads to more exploration and thus less chance to estimate the reward mean incorrectly. Things are unfortunately not that simple. The number of samples <img src="https://render.githubusercontent.com/render/math?math=T_i(t-1)"> in the  <img src="https://render.githubusercontent.com/render/math?math=UCB_i"> index is a random variable, so choosing the confidence level, at least naievely, should be done a bit smaller than <img src="https://render.githubusercontent.com/render/math?math=1/n">.

Page 86 might be ok to include into the motivation???

## Regret Analysis
> Theorem 7.1. The regret of the UCB algorithm shown above on any stochastic k-armed 1-subgaussian bandit problem, fro any horizon n, and  <img src="https://render.githubusercontent.com/render/math?math=\delta = 1/n^2"> is <img src="https://render.githubusercontent.com/render/math?math=R_n \leq 3\sum_{i=1}^{k}\Delta_i + \sum_{i:\Delta_i > 0} \frac{16\log(n)}{\Delta_i}">.

TODO: compare the regret with the explore-then-commit

The proof of the above theorem relies on the regret decomposition identify from [decomposition lemma](4_stochastic_bandits.md#decomposing-the-regret), <img src="https://render.githubusercontent.com/render/math?math=\sum_{i=1}^{k} \Delta_i \mathbb{E}[T_i(n)]"> 



Let's decouple the randomness from the behavior of the UCB algorithm and define <img src="https://render.githubusercontent.com/render/math?math=G_i"> as a "good" event by <img src="https://render.githubusercontent.com/render/math?math=G_i = \{u_1 < min_{t\in[n]}UCB_1(t, \delta)\} \cap \{\hat{u_i}_{u_i} + \sqrt{\frac{2}{u_i} \log (\frac{1}{\delta})} < \mu_1\}"> where <img src="https://render.githubusercontent.com/render/math?math=u_i \in [n]"> is a constant to be chosen later. <img src="https://render.githubusercontent.com/render/math?math=G_i"> is the event when the reward mean of the optimal arm <img src="https://render.githubusercontent.com/render/math?math=u_1"> is never underestimated by its upper confidence bound while at the same time the upper confidence bound for the mean of arm <img src="https://render.githubusercontent.com/render/math?math=i"> after <img src="https://render.githubusercontent.com/render/math?math=u_i"> pulls is below the mean reward of the optimal arm. 

TODO: definnovat mu_is

The theorem will be proven by bounding <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[T_i(n)] = \mathbb{E}[\mathbb{I}\{G_i\}T_i(n)] %2B \mathbb{E}[\mathbb{I} \{G_i^{\mathsf{c}}\}T_i(n)]"> for each suboptimal arm <img src="https://render.githubusercontent.com/render/math?math=i">.The proof is split into two parts.

**1. If <img src="https://render.githubusercontent.com/render/math?math=G_i"> occur, then <img src="https://render.githubusercontent.com/render/math?math=i"> will be played at most <img src="https://render.githubusercontent.com/render/math?math=u_i"> times, so that <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[\mathbb{I}\{G_i\}T_i(n)] \leq u_i">**

Let's show by contradiction that <img src="https://render.githubusercontent.com/render/math?math=T_i(n) \leq u_i"> when <img src="https://render.githubusercontent.com/render/math?math=G_i"> holds. Suppose that <img src="https://render.githubusercontent.com/render/math?math=T_i(n) > u_i">. Then arm <img src="https://render.githubusercontent.com/render/math?math=i"> was played more than <img src="https://render.githubusercontent.com/render/math?math=u_i"> times over the <img src="https://render.githubusercontent.com/render/math?math=n"> rounds, so there must exist a round <img src="https://render.githubusercontent.com/render/math?math=t \in [n]"> where <img src="https://render.githubusercontent.com/render/math?math=T_i(t-1)=u_i"> and <img src="https://render.githubusercontent.com/render/math?math=A_t=i">. The proof goes as follows
1. <img src="https://render.githubusercontent.com/render/math?math=UCB_i(t-1, \delta) = \hat{\mu_i}(t-1) + \sqrt{\frac{2\log(1/\delta)}{T_i(t-1)}}"> // given the definition of the  <img src="https://render.githubusercontent.com/render/math?math=UCB_i(t-1, \delta)">
1. <img src="https://render.githubusercontent.com/render/math?math== \hat{\mu_i}_{u_i} + \sqrt{\frac{2\log(1/\delta)}{u_i}}"> // since <img src="https://render.githubusercontent.com/render/math?math=T_i(t-1)=u_i"> as we stated above
1. <img src="https://render.githubusercontent.com/render/math?math=<\mu_1"> // given the definition of <img src="https://render.githubusercontent.com/render/math?math=G_i">
1. <img src="https://render.githubusercontent.com/render/math?math=<\UCB_1(t-1, \delta)"> // given the definition of <img src="https://render.githubusercontent.com/render/math?math=G_i">


| <!-- --> | <!-- --> | <!-- --> |
| :---         |     :---:      |          ---: |
| <img src="https://render.githubusercontent.com/render/math?math=UCB_i(t-1, \delta) = \hat{\mu_i}(t-1) + \sqrt{\frac{2\log(1/\delta)}{T_i(t-1)}}">   |     | definition of the  <img src="https://render.githubusercontent.com/render/math?math=UCB_i(t-1, \delta)">    |
| <img src="https://render.githubusercontent.com/render/math?math== \hat{\mu_i}_{u_i} + \sqrt{\frac{2\log(1/\delta)}{u_i}}">     |        | since <img src="https://render.githubusercontent.com/render/math?math=T_i(t-1)=u_i"> as we stated above     |
| <img src="https://render.githubusercontent.com/render/math?math=<\mu_1">     |        | given the definition of <img src="https://render.githubusercontent.com/render/math?math=G_i">    |
| <img src="https://render.githubusercontent.com/render/math?math=<\UCB_1(t-1, \delta)">     |        | given the definition of <img src="https://render.githubusercontent.com/render/math?math=G_i">    |

Since <img src="https://render.githubusercontent.com/render/math?math=UCB_i(t-1, \delta)"> is smaller than <img src="https://render.githubusercontent.com/render/math?math=UCB_1(t-1, \delta)">, <img src="https://render.githubusercontent.com/render/math?math=A_t = argmax_j UCB_j(t-1, \delta) \neq i">, which is a contradiction. This means that if <img src="https://render.githubusercontent.com/render/math?math=G_i"> occurs, then <img src="https://render.githubusercontent.com/render/math?math=T_i(n) \leq u_i">.  


 <div class="div-table">
    <div class="div-table-row">
          <div class="div-table-col_eq"><img src="https://render.githubusercontent.com/render/math?math=UCB_i(t-1, \delta) = \hat{\mu_i}(t-1) + \sqrt{\frac{2\log(1/\delta)}{T_i(t-1)}}"> </div>
        <div class="div-table-col_expl">definition of the  <img src="https://render.githubusercontent.com/render/math?math=UCB_i(t-1, \delta)"></div>
    </div>
    <div class="div-table-row">
        <div class="div-table-col_eq"><img src="https://render.githubusercontent.com/render/math?math== \hat{\mu_i}_{u_i} + \sqrt{\frac{2\log(1/\delta)}{u_i}}"></div>
        <div class="div-table-col_expl">since <img src="https://render.githubusercontent.com/render/math?math=T_i(t-1)=u_i"> as we stated above</div>
   </div>
    <div class="div-table-row">
        <div class="div-table-col_eq"><img src="https://render.githubusercontent.com/render/math?math=<\mu_1"></div>        
        <div class="div-table-col_expl">given the definition of <img src="https://render.githubusercontent.com/render/math?math=G_i"></div>
   </div>
</div>


**2. The complement event <img src="https://render.githubusercontent.com/render/math?math=G_i^{\mathsf{c}}"> occurs with low probability, so that <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[\mathbb{I} \{G_i^{\mathsf{c}}\}T_i(n)] = \mathbb{P}(G_i^{\mathsf{c}})n"> with <img src="https://render.githubusercontent.com/render/math?math=\mathbb{P}(G_i^{\mathsf{c}})"> being small**


 
     







If you have any questions or comments, I would be happy if you write them in the [discussion](https://github.com/azikoss/bandit_summaries/discussions/categories/6-explore-then-commit) section. 
 
# References
This text is *my* summary from the 7. Chapter of [Bandit Algorithm](https://tor-lattimore.com/downloads/book/book.pdf) book. The summary may contain copy&pasted text from the book. 
