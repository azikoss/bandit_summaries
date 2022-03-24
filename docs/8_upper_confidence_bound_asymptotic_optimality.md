# 8. The Upper Confidence Bound Algorithm: Asymptotic Optimality
This chapter introduces a simple yet very well functioning algorithm called the upper confidence bound (UCB) algorithm. The algorithm has multiple advantages over the [explore-then-commit (ETC) algorithm](6_explore_then_commit.md). It does not rely on the advanced knowledge of the suboptimality gaps and works well when there are more than two arms. The introduced UCB algorithm, just as ETC, depends on the horizon <img src="https://render.githubusercontent.com/render/math?math=n"> but the version presented in the next chapter does not. 


<img src="https://render.githubusercontent.com/render/math?math=\hat{\mu}=\frac{1}{n} \Sigma_{t=1}^{n}X_t">



If you have any questions or comments, I would be happy if you write them in the [discussion](https://github.com/azikoss/bandit_summaries/discussions/categories/7-ucb) section. 
 
# References
This text is *my* summary from the 8. Chapter of [Bandit Algorithm](https://tor-lattimore.com/downloads/book/book.pdf) book. The summary may contain copy&pasted text from the book. 
