# 8. The Upper Confidence Bound Algorithm: Asymptotic Optimality
<center>
<img width="480" src="./assets/8_ucb_asymptotic.png">
</center>

**The Upper Confidence Bound (UCB)** algorithm [introduced](7_upper_confidence_bound.md) in the previous chapter is **not anytime** as it requires advanced knowledge of the horizon <img src="https://render.githubusercontent.com/render/math?math=n">. This drawback is resolved in this chapter. 

The **asymptotically optimal UCB** introduced in this chapter differs from the previously introduced UCB algorithm just in the **choice of the confidence level** - dictated by the regret analysis. **The algorithm goes as follows**
1. Input: <img src="https://render.githubusercontent.com/render/math?math=k"> arms
1. Choose each arm once
1. Choose <img src="https://render.githubusercontent.com/render/math?math=A_t = \argmax_i(\hat{\mu}_i(t-1) %2B \sqrt{\frac{2\log{f(t)}}{T_i(t-1)}})"> where <img src="https://render.githubusercontent.com/render/math?math=f(t) = 1 %2B t\log^2(t)">

## Bounding the frequency of an index
Before the regret analysis, we introduce a lemma that **bounds the number of times the index** (such as UCB) of a suboptimal arm will be **larger** than some **threshold above its mean**. 
 
 > Lemma 8.2. Let <img src="https://render.githubusercontent.com/render/math?math=f(t) = X_1, ..., X_n"> be a sequence of independent 1-subgaussian random variables, <img src="https://render.githubusercontent.com/render/math?math=\hat{\mu}_t = \frac{1}{t}\sum_{s=1}^{t} X_s">, <img src="https://render.githubusercontent.com/render/math?math=\epsilon > 0">,  <img src="https://render.githubusercontent.com/render/math?math=a > 0">, and <img src="https://render.githubusercontent.com/render/math?math=\kappa = \sum_{t=1}^{n} \mathbb{1} \{\hat{\mu}_t %2B \sqrt{\frac{2a}{t}} \geq \epsilon\}">, <img src="https://render.githubusercontent.com/render/math?math=\kappa^' = u %2B \sum_{t=\lceil u \rceil}^{t=n} \mathbb{1} \{\hat{\mu}_t %2B \sqrt{\frac{2a}{t}} \geq \epsilon\}">, where <img src="https://render.githubusercontent.com/render/math?math=u=2a\epsilon^{-2}">. Then it holds that <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[\kappa] \leq \mathbb{E}[\kappa^'] \leq 1 %2B \frac{2}{\epsilon^2}(a + \sqrt{\pi a} %2B 1)">.

The proof goes as follows.
<div class="div-table">
    <div class="div-table-row">
        <div class="div-table-col_eq_wide_expl">
            <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[\kappa] \leq \mathbb{E}[\kappa^']">
        </div>
        <div class="div-table-col_expl_wide_expl">
            Since <img src="https://render.githubusercontent.com/render/math?math=X_i"> are
            independent 1-subgaussians with <img
                src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[\hat{\mu}_t] = 0">,
            then in expectation <img
                src="https://render.githubusercontent.com/render/math?math=\hat{\mu}_t %2B \sqrt{\frac{2a}{t}}">
            cannot be smaller than <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> until <img src="https://render.githubusercontent.com/render/math?math=t"> is at least <img src="https://render.githubusercontent.com/render/math?math=2a\epsilon^{-2}">. This is because when <img src="https://render.githubusercontent.com/render/math?math=t=u">, then
            <img src="https://render.githubusercontent.com/render/math?math=\sqrt{\frac{2a}{t}} = \epsilon">
            and all steps before it holds that <img src="https://render.githubusercontent.com/render/math?math=\sqrt{\frac{2a}{t}} \geq \epsilon">. If <img src="https://render.githubusercontent.com/render/math?math=u"> would be an integer, then <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[\kappa] = \mathbb{E}[\kappa^']">.                        
        </div>
    </div>
    <div class="div-table-row">
        <div class="div-table-col_eq_wide_expl">
            <img src="https://render.githubusercontent.com/render/math?math== u %2B \sum_{t=\lceil u \rceil}^{n} \mathbb{P} (\hat{\mu}_t %2B \sqrt{\frac{2a}{t}} \leq \epsilon)">
        </div>
        <div class="div-table-col_expl_wide_expl">
            indicator function was changed to probability because of the expected value
        </div>
    </div>
    <div class="div-table-row">
        <div class="div-table-col_eq_wide_expl">
            <img src="https://render.githubusercontent.com/render/math?math=\leq u %2B \sum_{t=\lceil u \rceil}^{n} \exp(-\frac{t(\epsilon-\sqrt{\frac{2a}{t}})^2}{2})">
        </div>
        <div class="div-table-col_expl_wide_expl">
            by <a href="5_concentration_of_measure.html#bounding-the-sample-reward-mean">bounding
            the
            tail behavior of the subgaussian</a> (with <img
                src="https://render.githubusercontent.com/render/math?math=\mu=0">)
        </div>
    </div>
    <div class="div-table-row">
        <div class="div-table-col_eq_wide_expl">
            <img src="https://render.githubusercontent.com/render/math?math=\leq 1 %2B u %2B \int_{u}^{\infinity} \exp(-\frac{t(\epsilon-\sqrt{\frac{2a}{t}})^2}{2}) \,dt">
        </div>
        <div class="div-table-col_expl_wide_expl">
              <img src="https://render.githubusercontent.com/render/math?math=1"> was added because the ceiling operator was removed from <img
                src="https://render.githubusercontent.com/render/math?math=u"> and the integral
            spanning from
            <img
                    src="https://render.githubusercontent.com/render/math?math=u"> to <img
                src="https://render.githubusercontent.com/render/math?math=\infinity">
            substituted the
            sum
            that goes only till <img
                src="https://render.githubusercontent.com/render/math?math=n">
        </div>
    </div>
    <div class="div-table-row">
        <div class="div-table-col_eq_wide_expl">
            <img src="https://render.githubusercontent.com/render/math?math== 1 %2B \frac{2}{\epsilon^2}(a %2B \sqrt{\pi a} %2B 1)">
        </div>
        <div class="div-table-col_expl_wide_expl">
            using Algebra
        </div>
    </div>
</div>

## Regret Analysis
Let's introduce and proof the theorem that bounds the introduced algorithm above.

> Theorem 8.1. For any 1-subgaussian bandit, the regret of the algorithm above satisfies <img src="https://render.githubusercontent.com/render/math?math=R_n \leq \sum_{i:\Delta_i > 0} \inf_{\epsilon \in (0, \Delta_i)} \Delta_i (1 %2B \frac{5}{\epsilon^2} %2B \frac{2(\log{f(n)} %2B \sqrt{\pi \log{f(n)}} %2B 1)}{(\Delta_i - \epsilon)^2})">.

The proof is build up on the [regret decomposition lemma](4_stochastic_bandits.md#decomposing-the-regret) <img src="https://render.githubusercontent.com/render/math?math=\sum_{i=1}^{k} \Delta_i \mathbb{E}[T_i(n)]"> and on bounding <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[T_i(n)]"> such that <img src="https://render.githubusercontent.com/render/math?math=T_i"> of a suboptimal arm <img src="https://render.githubusercontent.com/render/math?math=i"> is decomposed into two terms. The first measures the number of times the index of the optimal arm is less than <img src="https://render.githubusercontent.com/render/math?math=\mu_1 - \epsilon">. The second measures the number of times that suboptimal arm is played (<img src="https://render.githubusercontent.com/render/math?math=A_t=i">) and its index is larger than <img src="https://render.githubusercontent.com/render/math?math=\mu_1 - \epsilon">.

<div class="div-table">
    <div class="div-table-row">
        <div class="div-table-col_eq">
            <img src="https://render.githubusercontent.com/render/math?math=T_i(n) = \sum_{t=1}^{n} \mathbb{I}\{A_t = i\}">
        </div>
        <div class="div-table-col_expl">
        &nbsp;     
        </div>
    </div>
    <div class="div-table-row">
        <div class="div-table-col_eq">
            <img src="https://render.githubusercontent.com/render/math?math=\leq \sum_{t=1}^{n} \mathbb{I}\{\hat{\mu}_1(t-1) %2B \sqrt{\frac{2\log{f(t)}}{T_1(t-1)}} \leq \mu_1 - \epsilon\}"> <img src="https://render.githubusercontent.com/render/math?math=%2B \sum_{t=1}^{n} \mathbb{I}\{\hat{\mu}_i(t-1) %2B \sqrt{\frac{2\log{f(t)}}{T_i(t-1)}} \geq \mu_1 - \epsilon \: \textrm{and} \: A_t = i\}">   
        </div>
        <div class="div-table-col_expl">
        (8.4)     
        </div>
    </div>
</div>

Next we bound the expectation of each of the above sums. The first is bounded as follows.

<div class="div-table">
    <div class="div-table-row">
        <div class="div-table-col_eq">
            <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[\sum_{t=1}^{n} \mathbb{I}\{\hat{\mu}_1(t-1) %2B \sqrt{\frac{2\log{f(t)}}{T_1(t-1)}} \leq \mu_1 - \epsilon\}]"> 
        </div>
        <div class="div-table-col_expl">
        &nbsp;     
        </div>
    </div>
    <div class="div-table-row">
        <div class="div-table-col_eq">
          <img src="https://render.githubusercontent.com/render/math?math=\leq \sum_{t=1}^{n}\sum_{s=1}^{n} \mathbb{P}\{\hat{\mu}_{1s} %2B \sqrt{ \frac{2\log{f(t)}}{s} } \leq \mu_1 - \epsilon\}"> 
        </div>
        <div class="div-table-col_expl">
        union bound over all possible values of <img src="https://render.githubusercontent.com/render/math?math=T_1(t-1)">         
        </div>
    </div>
    <div class="div-table-row">
        <div class="div-table-col_eq">
          <img src="https://render.githubusercontent.com/render/math?math=\leq \sum_{t=1}^{n}\sum_{s=1}^{n} \exp(-\frac{s(\sqrt{\frac{2\log{f(t)}}{s}} %2B \epsilon)^2}{2})"> 
        </div>                    
        <div class="div-table-col_expl">
        by <a href="5_concentration_of_measure.html#bounding-the-sample-reward-mean">bounding the tail behavior of the subgaussian</a>                
        </div>
    </div>  
    <div class="div-table-row">
        <div class="div-table-col_eq">
          <img src="https://render.githubusercontent.com/render/math?math=\leq \sum_{t=1}^{n} \frac{1}{f(t)}\sum_{s=1}^{n} \exp(-\frac{s\epsilon^2}{2})"> 
        </div>                    
        <div class="div-table-col_expl">
        Using algebra within <img src="https://render.githubusercontent.com/render/math?math=\exp"> and leaving some terms out (from <img src="https://render.githubusercontent.com/render/math?math=\exp">)       
        </div>
    </div>  
      <div class="div-table-row">
        <div class="div-table-col_eq">
          <img src="https://render.githubusercontent.com/render/math?math=\leq \frac{5}{\epsilon^2}"> 
        </div>                    
        <div class="div-table-col_expl">
        Algebraic exercise. The function <img src="https://render.githubusercontent.com/render/math?math=f(t)"> was chosen so this bound would hold.            
        </div>
    </div>    
</div> 

The second term in (8.4) we use Lemma 8.2. 

<div class="div-table">
    <div class="div-table-row">
        <div class="div-table-col_eq">
            <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[\sum_{t=1}^{n} \mathbb{I}\{\hat{\mu}_i(t-1) %2B \sqrt{\frac{2\log{f(t)}}{T_i(t-1)}} \geq \mu_1 - \epsilon \: \textrm{and} \: A_t = i\}]">
        </div>
        <div class="div-table-col_expl">
        &nbsp;     
        </div>
    </div>
     <div class="div-table-row">
        <div class="div-table-col_eq">
            <img src="https://render.githubusercontent.com/render/math?math=\leq\mathbb{E}[ \sum_{t=1}^{n} \mathbb{I}\{\hat{\mu}_i(t-1) %2B \sqrt{\frac{2\log{f(n)}}{T_i(t-1)}} \geq \mu_1 - \epsilon \: \textrm{and} \: A_t = i\}]">
        </div>
        <div class="div-table-col_expl">
         <img src="https://render.githubusercontent.com/render/math?math=f(t)"> was replaced by <img src="https://render.githubusercontent.com/render/math?math=f(n)"> in the fraction as <img src="https://render.githubusercontent.com/render/math?math=f(t) \leq f(n)">     
        </div>
    </div>    
    <div class="div-table-row">
        <div class="div-table-col_eq">
            <img src="https://render.githubusercontent.com/render/math?math=\leq \mathbb{E}[\sum_{s=1}^{n} \mathbb{I}\{\hat{\mu}_{is} %2B \sqrt{\frac{2\log{f(n)}}{s}} \geq \mu_1 - \epsilon]"> 
        </div>
        <div class="div-table-col_expl">
         The UCB index is calculated for all steps <img src="https://render.githubusercontent.com/render/math?math=s \in [n]"> and not just steps when arm <img src="https://render.githubusercontent.com/render/math?math=i"> is pulled    
        </div>
    </div>        
    <div class="div-table-row">
        <div class="div-table-col_eq">
            <img src="https://render.githubusercontent.com/render/math?math== \mathbb{E}[\sum_{s=1}^{n} \mathbb{I}\{\hat{\mu}_{is} - \mu_i %2B \sqrt{\frac{2\log{f(n)}}{s}} \geq \Delta_i - \epsilon]"> 
        </div>
        <div class="div-table-col_expl">
        <img src="https://render.githubusercontent.com/render/math?math=\mu_1"> was substituted by <img src="https://render.githubusercontent.com/render/math?math=\mu_i - \Delta_i">     
        </div>
    </div>    
    <div class="div-table-row">
        <div class="div-table-col_eq">
            <img src="https://render.githubusercontent.com/render/math?math=\leq 1 %2B \frac{2}{(\Delta_i - \epsilon)^2}(\log{f(n)} %2B    \sqrt{\pi\log{f(n)}} %2B 1) ">
        </div>
        <div class="div-table-col_expl">
        by Lemma 8.2 where <img src="https://render.githubusercontent.com/render/math?math=\hat{u}_t = \hat{\mu}_{is} - \mu_i"> 
        </div>
    </div>    
</div>
   
The proof gets completed by substituting the results of two above bounds into (8.4).

# Comparison with the UCB algo from Chapter 7
Part of the Theorem 8.1 that was not mentioned yet is that the **worst-case bound** of the UCB algorithm introduced in this chapter is <img src="https://render.githubusercontent.com/render/math?math=R_n \leq C \sum_{i=0}^{k} \Delta_i %2B 2 \sqrt{C n  k \log(n)}">. This bound has the **same order** as the one of the **UCB algorithm introduced in Chapter 7** (<a href="7_upper_confidence_bound.html#regret-bound-without-suboptimality-gaps">Theorem 7.2</a>). The **leading term** that governs the asymptotic growth of the bound of the algorithm introduced in this chapter is however **smaller**, thus better. This improvement yields from making **the confidence interval slightly smaller** as we did not need to proof that <img src="https://render.githubusercontent.com/render/math?math=\hat{\mu}_{1s} \geq \mu_1"> for all <img src="https://render.githubusercontent.com/render/math?math=s"> with high probability but only that <img src="https://render.githubusercontent.com/render/math?math=\hat{\mu}_{1s} \geq \mu_1 - \epsilon"> for small <img src="https://render.githubusercontent.com/render/math?math=\epsilon">. 


If you have any questions or comments, I would be happy if you write them in the [discussion](https://github.com/azikoss/bandit_summaries/discussions/categories/8-ucb-asymptotic-optimality) section. 
 
# References
This text is *my* summary from the 8. Chapter of [Bandit Algorithm](https://tor-lattimore.com/downloads/book/book.pdf) book. The summary may contain copy&pasted text from the book. 
