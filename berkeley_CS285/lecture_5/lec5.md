## 0. Meta
- Course: CS 285, Reinforcement Learning
- Date: 2025.12.
- Lecturer: Sergey Levine
- Source/Link: (https://www.youtube.com/watch?v=GKoKNYaBvM0&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=15)

---

## 1. Big Picture (one-sentence summary of this lecture)
<!-- 오늘 강의의 핵심 메시지/주제 한두 줄 -->
- 

---

## 2. Key Concepts
<!-- 중요한 용어 / 개념 리스트업 (정확한 정의가 아니어도 됨, 나중에 수정 가능) -->
- $P(\tau)$ = $P_\theta(\tau\)$
- Monte Carlo (MC) estimation
    - By taking the sample mean of multiple draws, we can do a MC estimation
- Maximun Likelihood (ML) vs Policy Gradient (PG)
    - ML: Consider action of data $a$ as a correct answer and unconditionaly increase $\log \pi_\theta(a|s)$
    - PG: The sample action has no guarantee of a correct answer. Increasing probability of trajectory with big reward, reducing probability of trajectory with small reward.
- At the policy gradient, reducing variance is really important.
-  Marcov property (MC property) vs Causality
    - MC property: The sate in the future is independent of the state in the past givent the present. And MC property is sometimes true somtimes not depending on your temporal porcess
    - Causality: The policy at time $t^\prime$ cannot affect the reward at another time step $t$ if $t$ < $t^\prime$. And Causality is always true

- Natural Policy Gradient (Natural PG)
    - $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$
        - This is the original Vanila PG equation. It means Go one step to gradient direction from parameter space ($\theta$ space)
        - But the ploblem is a one step forward to parameter is not equal as one step forward to policy
        - A small change in $\sigma$ can cause a large change in the policy (distribution), whereas the same-sized change in k may have a much smaller effect
        - poor conditioning in the lecture means gradient of derection $\sigma$ become too big. so updating focus on decreasing $\sigma$

![Figure1](img/natural_dg.png)
1. Vanilla GD
    - The arrows don't point cleanly toward the optimun
    - The $\sigma$ component dominates the gradient, so the updates get pulled into "just reducing the $\sigma$" rather than moving straight toward the best parameters
2. Natural GD
    - The arrows converge nicely toward the optimum
    - Multiplying by $F^{-1}$ rescales/preconditions the gradient, so the step is measured in terms of "policy change" (distribution space) and ends up pointing in a more appropriate direction
    - $F^{-1}$ is an estimate of the fischer information matrix
    - So final equation is $\theta \leftarrow \theta + \alpha F^{-1}\nabla_\theta J(\theta)$

- Update
$$\Delta k=\alpha \frac{g_k}{F_{kk}},\qquad
\Delta \sigma=\alpha \frac{g_\sigma}{F_{\sigma\sigma}}
$$
- Vanilla: $\Delta k_{\text{van}}=\alpha g_k$
- Natural: $\Delta k_{\text{nat}}=\alpha \frac{g_k}{F_{kk}}$
---

## 3. Important Equations / Diagrams
<!-- 수식, 그림/도식 설명. 수식은 LaTeX로 적어두면 나중에 재사용하기 좋음 -->
- $b^\*=\frac{\mathbb{E}[g(\tau)^2\,r(\tau)]}{\mathbb{E}[g(\tau)^2]}$
    - The optimal baseline is the expected reward re-weighted by gradient magnitudes for each parameter

- on-policy policy gradient
$$\nabla_{\theta} J(\theta)
\approx
\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}
\nabla_{\theta}\log \pi_{\theta}(a_{i,t}\mid s_{i,t})\,\hat{Q}_{i,t}$$

- off-policy policy gradient
$$\[
\textbf{off-policy policy gradient:}\quad
\nabla_{\theta'} J(\theta')
\approx
\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}
\frac{\pi_{\theta'}(s_{i,t},a_{i,t})}{\pi_{\theta}(s_{i,t},a_{i,t})}\,
\nabla_{\theta'}\log \pi_{\theta'}(a_{i,t}\mid s_{i,t})\,\hat{Q}_{i,t}
\]$$
---

## 4. Main Logic / Algorithm Steps
<!-- 강의에서 설명한 절차, 알고리즘 흐름을 단계별로 정리 -->
- Policy gradient
![Figure2](img/policy_gradient.png)
    1. Approximate $J(\theta)$ to sample
    2. Approximate gradient to sample. -> Change expectation to sample mean
    3. update!
    
---

## 5. My Confusions & Clarifications
<!-- 강의 들을 때 헷갈린 것들 + 나중에 찾아보고 이해한 내용 -->
### 5.1 What I didn’t understand (at first)
- When estimating the expectation($J(\theta)$), using sample mean can have high variance. Is it fine to use the sample avarage?
- When derivating probability, why they change probability to log probability
- How $F^{-1}$ can do scailing?

### 5.2 What I found later (from web search, GPT, and books)
- When estimating the expectation($J(\theta)$), theusing sample mean can have high variance. Is it fine to use the sample avarage?
    - Estimating means unbiased+LLN (Law of large number)
    - The estimating which we use is $\hat J(\theta)=\frac{1}{N}\sum_{i=1}^N R(\tau_i)$. htis estimation is unbiased so everagly correct the answer.
    - When the N increase $\hat J(\theta)$ is converge to $J(\theta)$ because of the LLN
- When derivating probability, why they change probability to log probability
    - Original formula is $\nabla_\theta J(\theta)=\int \nabla_\theta p_\theta(\tau)\, r(\tau)\, d\tau$
    - $p_\theta(\tau)=p(s_1)\prod_t \pi_\theta(a_t|s_t)\,p(s_{t+1}|s_t,a_t)$ there is circumstance trasition
    - So when you use log-derivation trick. That formula become $\nabla_\theta \log p_\theta(\tau)=\nabla_\theta\Big(\log p(s_1)+\sum_t \log \pi_\theta(a_t|s_t)+\sum_t \log p(s_{t+1}|s_t,a_t)\Big)$. In this formula, only $\pi_\theta$ rely on the $\theta$
    - So it becomes model-free and circumstance disappear
- How $F^{-1}$ can do scailing?
$$F^{-1}\approx
\begin{pmatrix}
1/F_{kk} & 0\\
0 & 1/F_{\sigma\sigma}
\end{pmatrix}$$
- $F_{kk}$: How sensitive the policy is to changes in $k$
    - If $F_{kk}$ is large, a samll change in $k$ causes a large change in the plicy. so $1/F_{kk}$ is small and the update $\Delta k$ is scaled down.
- $F_{\sigma\sigma}$: how sensitive the policy/distribution is to changes in $\sigma$ (curvature/metric in the $\sigma$ direction).
	- If $F_{\sigma\sigma}$ is large, a small change in $\sigma$ causes a large change in the policy, so $1/F_{\sigma\sigma}$ is small and the update $\Delta \sigma$ is scaled down.


---
