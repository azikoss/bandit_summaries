# 7. The Upper Confidence Bound Algorithm
This chapter introduces a simple yet very well functioning algorithm called the upper confidence bound (UCB) algorithm. The algorithm has multiple advantages over the [explore-then-commit (ETC) algorithm](6_explore_then_commit.md). It does not rely on the advance knowledge of the suboptimality gaps and works well when there are more than two arms. The introduced UCB algorithm as ETC depends on the horizon <img src="https://render.githubusercontent.com/render/math?math=n"> but the version presented in the next chapter does not. 

## The Optimism Principle
The UCB algorithm follows the principle of **optimism in the face of uncertainty**. This principle states that one should act as if the environment is as nice as **plausibly possible**. To give an example, imagine visiting a new country and making a choice whether to try a local restaurant or a well-known multinational chain. You are uncertain about the food of the local restaurant. It could be equally great as equally bad - you do not know because you have never been there. Following the above principle would mean taking a optimistic opinion about the food in the local restaurant and trying it out. Afterwards, you would update your current knowledge about it and make more informed decision next time. 


Following the optimism principle in the context of bandits means estimating with high probability the mean reward higher than its true value. This over-estimate is called an **Upper Confidence Bound (UCB)**. The intuition why this leas to sublinear regret goes as follows. A suboptimal arm is played only when its upper confidence bound is larger than the upper confidence bound of the optimal arm. But this cannot happen too often because playing a suboptimal arm decrease its upper confidence bound that would eventually fall below the one of the optimal arm. 


Let's formalize the above intuition and define the upper confidence bound. Let <img src="https://render.githubusercontent.com/render/math?math=(X_t)_{t=1}^{n}"> be a sequence of independent 1-subgaussion random variables with mean <img src="https://render.githubusercontent.com/render/math?math=\mu"> and <img src="https://render.githubusercontent.com/render/math?math=\hat{\mu}=\frac{1}{n} \Sigma_{t=1}^{n}X_t">. By what we learnt in [Chapter 5](5_concentration_of_measure.md#bounding-the-sample-reward-mean),
 
 
 <div class="div-table">
    <div class="div-table-row">
    <div class="div-table-col_eq">
          <img src="https://render.githubusercontent.com/render/math?math=\mathbb{P}(\mu \geq \hat{\mu} %2B \sqrt{\frac{2\log(1/\delta)}{n}}) \leq \delta"> for all <img src="https://render.githubusercontent.com/render/math?math=\delta \in (0,1)">
     </div>
     <div class="div-table-col_expl">(7.1)</div>
    </div>
     
 </div> 
    
Since the learner makes a decision at time <img src="https://render.githubusercontent.com/render/math?math=t">, defining an upper confidence bound based on the above inequality requires making the terms <img src="https://render.githubusercontent.com/render/math?math=\hat{\mu}"> and <img src="https://render.githubusercontent.com/render/math?math=n"> to be dependent on <img src="https://render.githubusercontent.com/render/math?math=t">. At time step <img src="https://render.githubusercontent.com/render/math?math=t">, the learner has observed <img src="https://render.githubusercontent.com/render/math?math=T_i(t-1)"> samples from arm <img src="https://render.githubusercontent.com/render/math?math=i"> and received rewards from that arm with an empirical mean of <img src="https://render.githubusercontent.com/render/math?math=\hat{\mu_i}(t-1)">. Then a reasonable candidate for "as large as plausibly possible" for the unknown reward mean of the *i*th arm is 

<img src="https://render.githubusercontent.com/render/math?math=UCB_i(t-1, \delta)"> = <img src="https://render.githubusercontent.com/render/math?math=\infinity"> if <img src="https://render.githubusercontent.com/render/math?math=T_i(t-1) = 0"> 

or otherwise <img src="https://render.githubusercontent.com/render/math?math=UCB_i(t-1, \delta) = \hat{\mu}(t-1) + %2B \sqrt{\frac{2\log(1/\delta)}{T_i(t-1)}}">. 

The expression <img src="https://render.githubusercontent.com/render/math?math=\sqrt{\frac{2\log(1/\delta)}{T_i(t-1)}}"> is called  **confidence width** or **exploration bonus**. 

Now, we can state a version of the UCB algorithm as follows
1. **Input** <img src="https://render.githubusercontent.com/render/math?math=k"> and <img src="https://render.githubusercontent.com/render/math?math=\delta">
1. **for** <img src="https://render.githubusercontent.com/render/math?math=t"> and <img src="https://render.githubusercontent.com/render/math?math=t \in 1, ..., n"> **do**
1. &emsp; Choose action <img src="https://render.githubusercontent.com/render/math?math=A_t = argmax_i UCB_i(t-1, \delta)">
1. &emsp; Observe reward <img src="https://render.githubusercontent.com/render/math?math=X_t"> and update upper confidence bounds
1. **end for**

The above algorithm is an **index algorithm**. An index algorithm chooses the arm in each round that maximizes some value, called the **index**. For the UCB algorithm, the index of arm <img src="https://render.githubusercontent.com/render/math?math=i"> is <img src="https://render.githubusercontent.com/render/math?math=UCB_i(t-1, \delta)">. 

<img src="https://render.githubusercontent.com/render/math?math=\delta"> is a called the **confidence level** and it quantifies the degree of certainty. <img src="https://render.githubusercontent.com/render/math?math=\delta"> should be small enough to ensure optimism with high probability but not so large that the suboptimal arms would be explored too frequently. Choosing the confidence level will be done in future chapters. For now, the choice of this parameter is done based on the following considerations. If the confidence interval fails and the index of an optimal arm drops belows its true mean, then it could happen that the algorithm stops playing the optimal arm and suffers linear regret. This suggest choosing <img src="https://render.githubusercontent.com/render/math?math=\delta \approx 1/n"> so that playing during a larger horizon would mean less chance to suffer from this failure since the smaller value of <img src="https://render.githubusercontent.com/render/math?math=\delta"> leads to more exploration and thus less chance to estimate the reward mean incorrectly. Things are unfortunately not that simple. The number of samples <img src="https://render.githubusercontent.com/render/math?math=T_i(t-1)"> in the  <img src="https://render.githubusercontent.com/render/math?math=UCB_i"> index is a random variable, so choosing the confidence level, at least naievely, should be done a bit smaller than <img src="https://render.githubusercontent.com/render/math?math=1/n">.

Page 86 might be ok to include into the motivation???

## Regret Analysis
> Theorem 7.1. The regret of the UCB algorithm shown above on any stochastic k-armed 1-subgaussian bandit problem, for any horizon <img src="https://render.githubusercontent.com/render/math?math=n">, and <img src="https://render.githubusercontent.com/render/math?math=\delta = 1/n^2"> is <img src="https://render.githubusercontent.com/render/math?math=R_n \leq 3\sum_{i=1}^{k}\Delta_i + \sum_{i:\Delta_i > 0} \frac{16\log(n)}{\Delta_i}">

The proof of the above theorem relies on the [decomposition lemma](4_stochastic_bandits.md#decomposing-the-regret) <img src="https://render.githubusercontent.com/render/math?math=\sum_{i=1}^{k} \Delta_i \mathbb{E}[T_i(n)]">. 

TODO: definnovat mu_is

Let's assume without loss of generality that the first arm is optimal so <img src="https://render.githubusercontent.com/render/math?math=\mu_1 = \mu^*">. 

We decouple the randomness from the behavior of the UCB algorithm and define <img src="https://render.githubusercontent.com/render/math?math=G_i"> as a "good" event by <img src="https://render.githubusercontent.com/render/math?math=G_i = \{u_1 < min_{t\in[n]}UCB_1(t, \delta)\} \cap \{\hat{u_i}_{u_i} %2B \sqrt{\frac{2}{u_i} \log (\frac{1}{\delta})} < \mu_1\}"> where <img src="https://render.githubusercontent.com/render/math?math=u_i \in [n]"> is a constant to be chosen later. <img src="https://render.githubusercontent.com/render/math?math=G_i"> is the event when the reward mean of the optimal arm <img src="https://render.githubusercontent.com/render/math?math=u_1"> is never underestimated by its upper confidence bound while at the same time the upper confidence bound for the reward mean of arm <img src="https://render.githubusercontent.com/render/math?math=i"> after <img src="https://render.githubusercontent.com/render/math?math=u_i"> pulls is below the mean reward of the optimal arm. 

The theorem will be proven by bounding <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[T_i(n)] = \mathbb{E}[\mathbb{I}\{G_i\}T_i(n)] %2B \mathbb{E}[\mathbb{I} \{G_i^{\mathsf{c}}\}T_i(n)] \leq  u_i %2B \mathbb{P}(G_i^{\mathsf{c}})n"> for each suboptimal arm <img src="https://render.githubusercontent.com/render/math?math=i">.The proof is split into two parts.

**1. If <img src="https://render.githubusercontent.com/render/math?math=G_i"> occur, then <img src="https://render.githubusercontent.com/render/math?math=i"> will be played at most <img src="https://render.githubusercontent.com/render/math?math=u_i"> times, so that <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[\mathbb{I}\{G_i\}T_i(n)] \leq u_i">**

Let's show by contradiction that <img src="https://render.githubusercontent.com/render/math?math=T_i(n) \leq u_i"> when <img src="https://render.githubusercontent.com/render/math?math=G_i"> holds. Suppose that <img src="https://render.githubusercontent.com/render/math?math=T_i(n) > u_i">. Then arm <img src="https://render.githubusercontent.com/render/math?math=i"> was played more than <img src="https://render.githubusercontent.com/render/math?math=u_i"> times over the <img src="https://render.githubusercontent.com/render/math?math=n"> rounds, so there must exist a round <img src="https://render.githubusercontent.com/render/math?math=t \in [n]"> where <img src="https://render.githubusercontent.com/render/math?math=T_i(t-1)=u_i"> and <img src="https://render.githubusercontent.com/render/math?math=A_t=i">. Then

 <div class="div-table">
    <div class="div-table-row">
          <div class="div-table-col_eq"><img src="https://render.githubusercontent.com/render/math?math=UCB_i(t-1, \delta) = \hat{\mu_i}(t-1) %2B \sqrt{\frac{2\log(1/\delta)}{T_i(t-1)}}"> </div>
        <div class="div-table-col_expl">definition of the  <img src="https://render.githubusercontent.com/render/math?math=UCB_i(t-1, \delta)"></div>
    </div>
    <div class="div-table-row">
        <div class="div-table-col_eq"><img src="https://render.githubusercontent.com/render/math?math== \hat{\mu_i}_{u_i} %2B \sqrt{\frac{2\log(1/\delta)}{u_i}}"></div>
        <div class="div-table-col_expl">since <img src="https://render.githubusercontent.com/render/math?math=T_i(t-1)=u_i"> as we stated above</div>
   </div>
    <div class="div-table-row">
        <div class="div-table-col_eq"><img src="https://render.githubusercontent.com/render/math?math=<\mu_1"></div>        
        <div class="div-table-col_expl">given the definition of <img src="https://render.githubusercontent.com/render/math?math=G_i"></div>
   </div>
   <div class="div-table-row">
        <div class="div-table-col_eq"><img src="https://render.githubusercontent.com/render/math?math=<\UCB_1(t-1, \delta)"></div>        
        <div class="div-table-col_expl">given the definition of <img src="https://render.githubusercontent.com/render/math?math=G_i"></div>
   </div>
</div>
Since <img src="https://render.githubusercontent.com/render/math?math=UCB_i(t-1, \delta)"> is smaller than <img src="https://render.githubusercontent.com/render/math?math=UCB_1(t-1, \delta)">,
 then <img src="https://render.githubusercontent.com/render/math?math=A_t \neq i">, which is a contradiction.This means that if <img src="https://render.githubusercontent.com/render/math?math=G_i"> occurs, <img src="https://render.githubusercontent.com/render/math?math=T_i(n) \leq u_i">.  


**2. The complement event <img src="https://render.githubusercontent.com/render/math?math=G_i^{\mathsf{c}}"> occurs with low probability, so that <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[\mathbb{I} \{G_i^{\mathsf{c}}\}T_i(n)] = \mathbb{P}(G_i^{\mathsf{c}})n"> with <img src="https://render.githubusercontent.com/render/math?math=\mathbb{P}(G_i^{\mathsf{c}})"> being small.**


 By its definition <img src="https://render.githubusercontent.com/render/math?math=G_i^{\mathsf{c}} = \{\mu_1 \geq min_{t\in[n]}UCB_1(t, \delta)\} \cup \{\hat{\mu}_{iu_i} %2B \sqrt{\frac{2\log(1/\delta)}{u_i}} \geq \mu_1\}">. Let's decompose the first of these sets <img src="https://render.githubusercontent.com/render/math?math=\{\mu_1 \geq min_{t\in[n]}UCB_1(t, \delta)\}">

 <div class="div-table">    
    <div class="div-table-row">
    <div class="div-table-col_eq">
          <img src="https://render.githubusercontent.com/render/math?math=\subset \{ \mu_1 \geq min_{s\in[n]}\hat{\mu_{1s}} %2B \sqrt{\frac{2\log(1/\delta)}{s}}\}">  
    </div>
    <div class="div-table-col_expl">using the definition of <img src="https://render.githubusercontent.com/render/math?math=UCB_1(t, \delta)">; I am not sure why the both sets are not equals though.
    </div>
    </div>
    <div class="div-table-row">
    <div class="div-table-col_eq">
          <img src="https://render.githubusercontent.com/render/math?math=%3D\cup_{s\in[n]} \{ \mu_1 \geq \hat{\mu_{1s}} %2B \sqrt{\frac{2\log(1/\delta)}{s}}\}">
     </div>
    <div class="div-table-col_expl">
     </div>      
    </div>
</div>

Then, we can bound the <img src="https://render.githubusercontent.com/render/math?math=\mathbb{P}(\{\mu_1 \geq min_{t\in[n]}UCB_1(t, \delta)\})">

 <div class="div-table">
    <div class="div-table-row">
    <div class="div-table-col_eq">
          <img src="https://render.githubusercontent.com/render/math?math=\leq \mathbb{P}(\cup_{s\in[n]} \{ \mu_1 \geq \hat{\mu_{1s}} %2B \sqrt{\frac{2\log(1/\delta)}{s}}\})">  
    </div>
    <div class="div-table-col_expl">given the decomposition above
    </div>
    </div>
    <div class="div-table-row">
    <div class="div-table-col_eq">
       <img src="https://render.githubusercontent.com/render/math?math=\leq \sum_{s=1}^{n}\mathbb{P}(\cup_{s\in[n]} \{ \mu_1 \geq \hat{\mu_{1s}} %2B \sqrt{\frac{2\log(1/\delta)}{s}}\})">
    </div>
    <div class="div-table-col_expl">given by summing independent subgaussian random variables</div>
    </div>    
    <div class="div-table-row">
    <div class="div-table-col_eq">
       <img src="https://render.githubusercontent.com/render/math?math=\leq n \delta">
    </div>
    <div class="div-table-col_expl">
    given by (7.1) 
    </div>
    </div>    
</div>

Let's bound the probability of the second set in <img src="https://render.githubusercontent.com/render/math?math=G_i^{\mathsf{c}}">. Let's assume that
 
 <div class="div-table">
    <div class="div-table-row">
    <div class="div-table-col_eq">
          <img src="https://render.githubusercontent.com/render/math?math=\delta_i - \sqrt{\frac{2\log(1/\delta)}{u_i}} \geq c\delta_i">
    </div>
    <div class="div-table-col_expl">(7.8)
    </div>
   </div>
 </div>

where <img src="https://render.githubusercontent.com/render/math?math=c\in (0,1)"> is a constant to be chosen later. Then, <img src="https://render.githubusercontent.com/render/math?math=\mathbb{P}(\hat{\mu}_{iu_i} %2B  \sqrt{\frac{2\log(1/\delta)}{u_i}} \geq \mu_1\))">


 <div class="div-table">
    <div class="div-table-row">
    <div class="div-table-col_eq">
          <img src="https://render.githubusercontent.com/render/math?math== \mathbb{P}(\hat{\mu}_{iu_{i}} - \mu_i \geq \delta_i - \sqrt{\frac{2\log(1/\delta)}{u_i}})">
    </div>
    <div class="div-table-col_expl">
    since  <img src="https://render.githubusercontent.com/render/math?math=\mu_1 = \mu_i %2B \delta_i">
    </div>
   </div>
   
   <div class="div-table-row">
    <div class="div-table-col_eq">
          <img src="https://render.githubusercontent.com/render/math?math=\leq \mathbb{P}(\hat{\mu}_{iu_{i}} - \mu_i \geq c\Delta_i)">
    </div>
    <div class="div-table-col_expl">
    by (7.8)
    </div>
   </div>
   
   <div class="div-table-row">
    <div class="div-table-col_eq">        
          <img src="https://render.githubusercontent.com/render/math?math=\leq \exp(-\frac{u_ic^2\Delta_i^2}{2})">
    </div>
    <div class="div-table-col_expl">   
    by this <a href="5_concentration_of_measure.html#bounding-the-sample-reward-mean">proof</a>
    </div>
   </div>   
 </div>
 
 
 Putting everything together yields the following bound <img src="https://render.githubusercontent.com/render/math?math=\mathbb{P}(G_i^{\mathsf{c}}) \leq n\delta %2B \exp(-\frac{u_ic^2\Delta_i^2}{2})">. Thus, the upper bound of 
 
 <div class="div-table">
    <div class="div-table-row">
    <div class="div-table-col_eq">
          <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[T_i(n)] \leq u_i %2B n(n\delta %2B \exp(-\frac{u_ic^2\Delta_i^2}{2}))">
    </div>
    <div class="div-table-col_expl">
    (7.9)
    </div>
   </div>
 </div>
 
 
 Now, we have to choose <img src="https://render.githubusercontent.com/render/math?math=u_i \in [n]"> such that (7.8) is satisfied. A natural choice is to the smallest integer (we want to pull the suboptimal arm the least times possible) for which (7.8) holds. By expressing <img src="https://render.githubusercontent.com/render/math?math=u_i"> from (7.8) and ceiling it to the nearest integer, we get <img src="https://render.githubusercontent.com/render/math?math=u_i = \lceil\frac{2\log(1/\delta)}{(1-c)^2\Delta_i^{2}}\rceil">. Using this choice of <img src="https://render.githubusercontent.com/render/math?math=u_i"> and the assumption that <img src="https://render.githubusercontent.com/render/math?math=\delta=1/n^2"> leads via (7.9) to 
 
 <div class="div-table">
    <div class="div-table-row">
    <div class="div-table-col_eq">
          <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[T_i(n)] \leq u_i %2B 1 %2B n^{1-2c^2/(1-c)^2}">
    </div>
    <div class="div-table-col_expl">by using <img src="https://render.githubusercontent.com/render/math?math=\delta=1/n^2">
    </div>
   </div>   
   <div class="div-table-row">
    <div class="div-table-col_eq">
          <img src="https://render.githubusercontent.com/render/math?math==\lceil\frac{2\log(1/\delta)}{(1-c)^2\Delta_i^{2}}\rceil %2B 1 %2B n^{1-2c^2/(1-c)^2}">
    </div>
    <div class="div-table-col_expl">(7.10)
    </div>
   </div>   
 </div>
 
 It remains to choose <img src="https://render.githubusercontent.com/render/math?math=c \in (0,1)">. The term <img src="https://render.githubusercontent.com/render/math?math=n^{1-2c^2/(1-c)^2}"> term from (7.10) will polynomially dependent on <img src="https://render.githubusercontent.com/render/math?math=n"> unless <img src="https://render.githubusercontent.com/render/math?math=2c^2/(1-c)^2 \geq 1">. Choosing <img src="https://render.githubusercontent.com/render/math?math=c"> too close to 1 would blow up the first term of (7.10). Taking the above into consideration, <img src="https://render.githubusercontent.com/render/math?math=c"> was chosen somewhat arbitrarily to <img src="https://render.githubusercontent.com/render/math?math=c=1/2">, which leads to <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[T_i(n)] \leq 3 %2B \frac{16\log{n}}{\delta_i^{2}}">. The proof is completed by substituting the above into (7.4).

The Theorem 7.1 depends on the knowledge of the suboptimality gaps that are not known in practice. This is addressed by the following theorem. 

> Theorem 7.2. The regret of UCB (defined by the algorithm above) on any stochastic k-armed 1-subgaussion bandit and and when <img src="https://render.githubusercontent.com/render/math?math=\delta = 1/n^2">, is bounded by 
><img src="https://render.githubusercontent.com/render/math?math=R_n \leq 8\sqrt{nk\log{(n)}} %2B 3 \sum_{i=1}^{k}\Delta_i">

The proof goes as follows

 <div class="div-table">
   <div class="div-table-row">
        <div class="div-table-col_eq">
        <img src="https://render.githubusercontent.com/render/math?math=R_n = \sum_{i=1}^{k}\Delta_i\mathbb{E}[T_i(n)] ">              
        </div>
        <div class="div-table-col_expl">
        by the regret decomposition lemma        
        </div>
   </div>
   
   <div class="div-table-row">
        <div class="div-table-col_eq">
        <img src="https://render.githubusercontent.com/render/math?math==\sum_{i: \Delta_i < \Delta}\Delta_i\mathbb{E}[T_i(n)] %2B \sum_{i: \Delta_i \geq \Delta}\Delta_i\mathbb{E}[T_i(n)]">                        
        </div>
        <div class="div-table-col_expl">  
        by splitting the sum into the sum where the suboptimality gaps are lower than some constant <img src="https://render.githubusercontent.com/render/math?math=\delta > 0"> to be tuned later and to the sum where the optimality gaps are larger than <img src="https://render.githubusercontent.com/render/math?math=\delta">       
        </div>
   </div>
   
   <div class="div-table-row">
        <div class="div-table-col_eq">      
        <img src="https://render.githubusercontent.com/render/math?math=\leq n\Delta %2B \sum_{i: \Delta_i \geq \Delta}\Delta_i\mathbb{E}[T_i(n)]">               
        </div>
        <div class="div-table-col_expl">        
        since <img src="https://render.githubusercontent.com/render/math?math=\sum_{i: \Delta_i < \Delta}T_i(n) \leq n"> or in words, the sum of the pulls can never be larger that the horizon <img src="https://render.githubusercontent.com/render/math?math=n"> and since <img src="https://render.githubusercontent.com/render/math?math=\Delta_i < \Delta">           
        </div>
   </div>
   
   <div class="div-table-row">
        <div class="div-table-col_eq">   
        <img src="https://render.githubusercontent.com/render/math?math=\leq n\Delta %2B \sum_{i: \Delta_i \geq \Delta}3\Delta_i %2B \frac{16\log{(n)}}{\Delta_i}">                     
        </div>
        <div class="div-table-col_expl">        
        given that <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[T_i(n)] \leq 3 %2B \frac{16\log{n}}{\delta_i^{2}}"> 
        </div>
   </div>
   
   <div class="div-table-row">
        <div class="div-table-col_eq">  
         <img src="https://render.githubusercontent.com/render/math?math=\leq n\Delta %2B \frac{16k\log(n)}{\Delta} %2B 3 \sum_{i=1}^{k}\Delta_i">                    
        </div>
        <div class="div-table-col_expl">
        given that there is at most <img src="https://render.githubusercontent.com/render/math?math=k"> arms with
          <img src="https://render.githubusercontent.com/render/math?math=\Delta_i \geq \Delta"> and given <img src="https://render.githubusercontent.com/render/math?math=\Delta_i \geq \Delta">              
        </div>
   </div>   
    <div class="div-table-row">
        <div class="div-table-col_eq">
        <img src="https://render.githubusercontent.com/render/math?math== 8 \sqrt{nk\log(n)} %2B %2B 3 \sum_{i=1}^{k}\Delta_i ">              
        </div>
        <div class="div-table-col_expl">   
        by choosing <img src="https://render.githubusercontent.com/render/math?math=\Delta = \sqrt{16k\log(n)/n}">     
        </div>
   </div>
 </div>

The above bound still includes the suboptimality gaps <img src="https://render.githubusercontent.com/render/math?math=\Delta_i">. This is however unavoidable because any reasonable algorithm must play each arm at least once. In any case, the term <img src="https://render.githubusercontent.com/render/math?math=3 \sum_{i=1}^{k}\Delta_i "> does not grow with the horizon and is thus negligible. 


TODO: compare the regret with the explore-then-commit
CONT --> implementation. 
      





If you have any questions or comments, I would be happy if you write them in the [discussion](https://github.com/azikoss/bandit_summaries/discussions/categories/6-explore-then-commit) section. 
 
 
 <div class="div-table">
   <div class="div-table-row">
        <div class="div-table-col_eq">              
        </div>
        <div class="div-table-col_expl">        
        </div>
   </div>            
 </div>
# References
This text is *my* summary from the 7. Chapter of [Bandit Algorithm](https://tor-lattimore.com/downloads/book/book.pdf) book. The summary may contain copy&pasted text from the book. 
