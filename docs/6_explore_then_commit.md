# 6. Explore-Then-Commit Algorithm
Explore-then-commit (ETC) is a simple algorithm that explores each arm a fixed number of times and then commit to playing that arm that performed the best during exploration.

ETC is determined by the number of times that each arm is explored, denoted as m. Since there are k arms and each arm is explored m times, the algoirthms explores mk times in total. Let <img src="https://render.githubusercontent.com/render/math?math=\hat{u}_i(t)"> be the avarage reward recieved from arm <img src="https://render.githubusercontent.com/render/math?math=i"> after round <img src="https://render.githubusercontent.com/render/math?math=t">, which is formally denoted as 
<img src="https://render.githubusercontent.com/render/math?math=\hat{u}_i(t) = \frac{1}{T_i(t)}\sum_{s=1}^{t}\mathbb{I}\{A_s = i \}X_s">, where <img src="https://render.githubusercontent.com/render/math?math=T_i(t) = \sum_{s=1}^{t}\mathbb{I}\{A_s=i\}"> expresses the number of times action <img src="https://render.githubusercontent.com/render/math?math=i"> has been played after round <img src="https://render.githubusercontent.com/render/math?math=t">. The ETC policy goes as follows
<table>               
<tr><td>
1. Input: m

2. In round <img src="https://render.githubusercontent.com/render/math?math=t"> choose action <img src="https://render.githubusercontent.com/render/math?math=A_t"> as

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://render.githubusercontent.com/render/math?math=(t \mod k) \%2B 1"> if <img src="https://render.githubusercontent.com/render/math?math=t \leq mk">

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://render.githubusercontent.com/render/math?math=\argmax_i\hat{\mu_i}(mk)"> if <img src="https://render.githubusercontent.com/render/math?math=t > mk"> 
</tr></td>
</table>

## Regret
Recall that <img src="https://render.githubusercontent.com/render/math?math=u_i"> is the true mean reward of action <img src="https://render.githubusercontent.com/render/math?math=i"> and <img src="https://render.githubusercontent.com/render/math?math=\Delta_i = \mu* - \mu_i"> is the suboptimality gap between the mean of the arm with the highest reward and arm <img src="https://render.githubusercontent.com/render/math?math=\i">.

> When ETC is interacting with any 1-subgaussion bandit (and <img src="https://render.githubusercontent.com/render/math?math=1 \leq m \leq n/k)">, <img src="https://render.githubusercontent.com/render/math?math=R_n \leq m \sum_{i=1}^{k}\Delta_i \%2B (n - mk)\sum_{i=1}^{k}\Delta_i \exp(-\frac{m\Delta_i^2}{4})"> 

Let's note that the above inequility can be adjusted to work when the reward distribution of all arms is not 1-subgaussiaons. Using 1-subgaussion is just for the sake of not writting the subgaussion constant <img src="https://render.githubusercontent.com/render/math?math=\sigma"> over and over. We (ETC) algorithm however relies on knowledge of the subgaussion parameter <img src="https://render.githubusercontent.com/render/math?math=\sigma">. 

The proof of the above theorem goes as follows
1. By the regret decomposition lemma from Chapter 4 (add bookmark), regret of any bandit algorithm can be written as <img src="https://render.githubusercontent.com/render/math?math=\sum_{a \in A} \Delta_a \mathbb{E}[T_a(n)]">.
1. For ETC, <img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[T_i(n)] = m \%2B (n-mk)\mathbb{P}(A_{mk %2B 1} = i)"> which reflects that each arm is played <img src="https://render.githubusercontent.com/render/math?math=m"> times during the exploration round in the remaining <img src="https://render.githubusercontent.com/render/math?math=n - mk"> rounds each arm is played with a certain probability.

The probability <img src="https://render.githubusercontent.com/render/math?math=\mathbb{P}(A_{mk %2B 1} = i)"> of playing arm <img src="https://render.githubusercontent.com/render/math?math=i"> during the exploitation

{:start="3"}
1. <img src="https://render.githubusercontent.com/render/math?math=\leq \mathbb{P}(\hat{\mu}_i(mk) \geq \max_{j \neq i} \hat{\mu_j}(mk))"> // this probability reflects the condition for playing arm <img src="https://render.githubusercontent.com/render/math?math=i"> during exploitation by the ETC algorithm. An upper bound is used in case when multiple arms are at time <img src="https://render.githubusercontent.com/render/math?math=mk"> the highest sample reward mean.  
1. <img src="https://render.githubusercontent.com/render/math?math=\leq \mathbb{P}(\hat{\mu}_i(mk) \geq \hat{\mu_1}(mk))"> when pressuming that arm <img src="https://render.githubusercontent.com/render/math?math=i"> is the optimal so that <img src="https://render.githubusercontent.com/render/math?math=\mu_1=\mu*=\max_i\mu_i">
1. <img src="https://render.githubusercontent.com/render/math?math== \mathbb{P}(\hat{\mu}_i(mk) - \mu_i - (\hat{\mu_1}(mk) - \mu_1) \geq \Delta_i)"> // <img src="https://render.githubusercontent.com/render/math?math=\Delta_i"> was added to the right hand side within the probability and <img src="https://render.githubusercontent.com/render/math?math=\mu_1 - \mu_i">, which equals to <img src="https://render.githubusercontent.com/render/math?math=\Delta_i">, to the left hand side.
1. <img src="https://render.githubusercontent.com/render/math?math=\leq \exp(-\frac{m\Delta_{i}^2}{4})"> since <img src="https://render.githubusercontent.com/render/math?math=\hat{\mu}_i(mk) - \mu_i - (\hat{\mu_1}(mk) - \mu_1)"> is <img src="https://render.githubusercontent.com/render/math?math=\sqrt{2/m}">-subgaussian and given the application of the theorem that bounds subgaussians (<img src="https://render.githubusercontent.com/render/math?math=\mathbb{E}[e^{\lambda X}] \leq e^{\lambda^2 \sigma^2 / 2}">). The above term is <img src="https://render.githubusercontent.com/render/math?math=\sqrt{2/m}">-subgaussian because <img src="https://render.githubusercontent.com/render/math?math=\hat{\mu}_i(mk) - \mu_i"> and <img src="https://render.githubusercontent.com/render/math?math=(\hat{\mu_1}(mk) - \mu_1)"> are both <img src="https://render.githubusercontent.com/render/math?math=1/\sqrt{m}">-subgaussians and their sum is <img src="https://render.githubusercontent.com/render/math?math=\sqrt{2/m}">-subgaussian (this follows the proof from chapter 5)

Substituting the row 4 into row 2 









If you have any questions or comments, I would be happy if you write them in the [discussion](https://github.com/azikoss/bandit_summaries/discussions/categories/5-concentration-of-measure) section. 

# References
This text is *my* summary from the 5. Chapter of [Bandit Algorithm](https://tor-lattimore.com/downloads/book/book.pdf) book. The summary contains copy&pasted text from the book as well as some additional text. 
