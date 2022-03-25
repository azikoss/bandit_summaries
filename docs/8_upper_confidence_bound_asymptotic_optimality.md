# 8. The Upper Confidence Bound Algorithm: Asymptotic Optimality
The Upper Confidence Bound (UCB) algorithm [introduced](7_upper_confidence_bound.md) in the previous chapter is not anytime as it requires advanced knowledge of the horizon <img src="https://render.githubusercontent.com/render/math?math=n">. This drawback is resovled in this chapter. 

The asymptotically optimally UCB differs from the previously introduced UCB algorithm just in the choice of the confidence level - dictated by the regret analysis. The algorihm goes as follows
1. Input: <img src="https://render.githubusercontent.com/render/math?math=k"> arms
1. Choose each arm once
1. Choose <img src="https://render.githubusercontent.com/render/math?math=A_t = \argmax_i(\hat{\mu}_i(t-1) %2B \sqrt{\frac{2\log{f(t)}}{T_i(t-1)}})"> where <img src="https://render.githubusercontent.com/render/math?math=f(t) = 1 %2B t\log^2(t)">

 ## Regret Analysis
 Before the regret analysis, we introduce a lemma that bounds the number of times the index (such as UCB) of a suboptimal arm will be larger than some threshold above its mean. 
 
 > Lemma 8.2. Let <img src="https://render.githubusercontent.com/render/math?math=f(t) = X_1, ..., X_n"> be a sequence of independent 1-subgaussian random variables, <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{t}\sum{s=1}_{t} X_s">, <img src="https://render.githubusercontent.com/render/math?math=\epsilon > 0">,  <img src="https://render.githubusercontent.com/render/math?math=a > 0">, and <img src="https://render.githubusercontent.com/render/math?math=\kappa = \sum_{n}^{t=1} \mathbb{1} \{\hat{\mu}_t %2B \sqrt{\frac{2a}{t}} \geq \epsilon\}">, <img src="https://render.githubusercontent.com/render/math?math=\kappa\prime = u %2B \sum_{t=\lceil u \rceil}^{t=n} \mathbb{1} \{\hat{\mu}_t %2B \sqrt{\frac{2a}{t}} \geq \epsilon\}">, where <img src="https://render.githubusercontent.com/render/math?math=u=2a\epsilon^-2">. Then it holds that <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[\kappa] \leq \mathbb{E}[\kappa\prime] \leq 1 %2B \frac{2}{\epsilon^2}(a + \sqrt{\pi a} %2B 1)">.

The proof goes as follows.
<div class="div-table">
    <div class="div-table-row">
        <div class="div-table-col_eq">
            <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[\kappa] \leq \mathbb{E}[\kappa\prime]">
        </div>
        <div class="div-table-col_expl">
            Since <img src="https://render.githubusercontent.com/render/math?math=X_i"> are
            independent 1-subgaussians with <img
                src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[\hat{\mu}_t] = 0">,
            then in expectation <img
                src="https://render.githubusercontent.com/render/math?math=\hat{\mu}_t %2B \sqrt{\frac{2a}{t}}">
            cannot be smaller than <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> until <img src="https://render.githubusercontent.com/render/math?math=t"> is at least <img src="https://render.githubusercontent.com/render/math?math=2a/\epsilon^2">. Note
            that when <img src="https://render.githubusercontent.com/render/math?math=t=u">, then
            <img src="https://render.githubusercontent.com/render/math?math=\sqrt{\frac{2a}{t}} = \epsilon">
            so all steps before <img src="https://render.githubusercontent.com/render/math?math=\sqrt{\frac{2a}{t}} > \epsilon">.
        </div>
    </div>
    <div class="div-table-row">
        <div class="div-table-col_eq">
            <img src="https://render.githubusercontent.com/render/math?math== u %2B \sum_{t=\lceil u \rceil}^{t=n} \mathbb{P} (\hat{\mu}_t %2B \sqrt{\frac{2a}{t}})">
        </div>
        <div class="div-table-col_expl">
            indicator function was changed to probability because of the expected value
        </div>
    </div>
    <div class="div-table-row">
        <div class="div-table-col_eq">
            <img src="https://render.githubusercontent.com/render/math?math=\leq u %2B \sum_{t=\lceil u \rceil}^{t=n} \exp(-\frac{t(\epsilon-\sqrt{\frac{2a}{t}})^2}{2})">
        </div>
        <div class="div-table-col_expl">
            by <a href="5_concentration_of_measure.html#bounding-the-sample-reward-mean">bounding
            the
            tail behavior of the subgaussian</a> (with <img
                src="https://render.githubusercontent.com/render/math?math=\mu=0">)
        </div>
    </div>
    <div class="div-table-row">
        <div class="div-table-col_eq">
            <img src="https://render.githubusercontent.com/render/math?math=\leq 1 %2B u %2B \int_{u}^{\infinity} \exp(-\frac{t(\epsilon-\sqrt{\frac{2a}{t}})^2}{2}) \,dt">
        </div>
        <div class="div-table-col_expl">
            the 1 was added because of the ceiling operator was removed from <img
                src="https://render.githubusercontent.com/render/math?math=u"> and the integral
            from
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
        <div class="div-table-col_eq">
            <img src="https://render.githubusercontent.com/render/math?math== 1 %2B \frac{2}{\epsilon^2}(a %2B \sqrt{\pi a} %2B 1)">
        </div>
        <div class="div-table-col_expl">
            using Algebra
        </div>
    </div>
</div>

If you have any questions or comments, I would be happy if you write them in the [discussion](https://github.com/azikoss/bandit_summaries/discussions/categories/7-ucb) section. 
 
# References
This text is *my* summary from the 8. Chapter of [Bandit Algorithm](https://tor-lattimore.com/downloads/book/book.pdf) book. The summary may contain copy&pasted text from the book. 
