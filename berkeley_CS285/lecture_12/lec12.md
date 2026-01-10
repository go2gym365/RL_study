## 0. Meta
- Course: CS285, Deep Reinforcement Learning
- Date: 2026.01.09
- Lecturer: Sergey Levine
- Source/Link: https://www.youtube.com/watch?v=UQGS4ycGv8g&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=50&pp=iAQB

## Part 1
- Limitation → Fix<!--이전 파트 한계 → 이번 파트 해결책-->
    - prev: Model-based RL v1.5 (MPC) replans every step, which helps with model error, but each plan is still open-loop (optimizes an action sequence). So they optimize an acton sequence and don't account for the fact that you'll get future observations and can react later
    - fix: Change toward the closed-loop setting: optimize a policy $\pi(a|s)$ instead of action sequence. 
        - MBRL v2.0: Differentiate through the learned dynamics and rewards to train the policy directly
        - But it isn't working well because of gradients exploding/vanishing. So usually use model as a simulator not a differentiable funciton to make synthetic data.
- Professor’s Emphasis <!--(핵심 한 줄): 강의에서 교수님이 ‘딱 박아준’ 문장-->
    - Backpropagating through long chains of learned dynamics and policy is like naively training an RNN with BPTT: gradient tend to explode or vanish
- Flow of this part <!--(어떤 흐름으로 강의가 진행되는지, 강의의 흐름을 반영하는건 맞지만 어떻게 알고리즘을 발전해나가는지 정리)-->
    1. MPC (v1.5)
        - collect data → train dynamics model → plan action sequence → execute first action → add transition to buffer → replan every step → periodically retrain the model.
    2. problem
        - Random shooting/CEM optimize and action sequence commited in advance
        - prof said about math test, MPC think that we can dicide the answer after check what the problem is. But it won't take a test because it'll decide right now but I might wrong because I don't know what's the problem is. 
        - So MPC doesn't replan like a open-loop
        - Conclusion: Optimize a policy that maps observed state to actions, matching the original RL objective
    3. Differentiate through the model (v2.0) and limitation
    ![Figure1](img/v2.0.png)
        - Build a computation graph with policy $\pi_\theta$, learned dynamics $f$, and reward $r$, then backprop through time to improve $\theta$
        - Early actions have compounding effects → huge gradients early, tiny gradients late → ill-conditioned optimization
        - Long chains multiply many Jacobians → exploding/vanishing gradients (same pathology as naive RNN BPTT)
        - LSTM-style fixes aren’t available because you can’t choose the real dynamics; $f$ must match the environment, which may be highly curved
    4. Use the model to simulate more synthetic samples (experience) and apply derivative-free (model-free) RL methods on that data
- Terminology Map <!--(용어 등치/정의)-->
    - Open-loop: Optimize an action sequence without conditioning of futer observation
    - Closed-loop: Optimize a policy $\pi(a|s)$ that reacts to the current state/observation
- Why it matters <!--(왜 중요한가 1~2줄) 이걸 놓치면 다음 파트가 왜 나오는지 이해가 안 됨-->
    - In this part, prof said that why the most obvious approacth end-to-end backpropagation through a learned model often fails, motivating why many practical model-based methods instead use models to generate data and then run model-free RL
## Part 2
- Limitation → Fix<!--이전 파트 한계 → 이번 파트 해결책-->
    - prev: Differentiate through the model tries to bckprop through the learned dynamics over long horizons. But the pathwise (backprop)gradient  contains a long product of Jacobians, causing exploding or vanishing gradients problems and severe ill-conditioning
    - fix
        1. Use policy gradient instead of pathwise gradients (likelihood-ratio/REINFORCE): those r not include Jacobian products
        2. MBRL (v2.5) use the model generate rollouts, then do PG updates from sampled trajectories
        3. But MBRL still fails with long model rollouts
        4. Use short model rollouts branched from real states
- Professor’s Emphasis <!--(핵심 한 줄): 강의에서 교수님이 ‘딱 박아준’ 문장-->
    - Policy gradient is a gradient estimator and can be used for any stochastic computation graph not only RL!
    - Long model rollouts are the enemy
- Flow of this part <!--(어떤 흐름으로 강의가 진행되는지, 강의의 흐름을 반영하는건 맞지만 어떻게 알고리즘을 발전해나가는지 정리)-->
    1. Two gradient method
        - Likelihood-ratio/policy gradient: gradient estimator which uses samples and doesn't depend on transition derivatives
        - Pathwise/backprop gradient: Chain rull differentiation through the dynamics and rewards. In here introduces Jacobian products
    2. Why pathwise through model is bad
        - Backprop through time multiplies lots of chain rule and cause explode/vanish like naive RNN BPTT
        - Second-order help like LQR in trajectory optimization isn't convenient because policy parameters couple all time steps
    3. Model based RL via model-free gradients (v2.5)
        - Collect real data → learn dynamics model → "sample many trajectories" in the model with current policy → update policy using policy gradient / actor-critic → repeat model-sampling + PG steps many times (without new real data) → occasionally go back to real env to gather more data and retrain model
        - Policy gradients aboid Jacobian products and model sampling makes more samples
    4. Limitation of v2.5
    ![Figure2](img/limitation_of_v2.5.png)
        - Same phenomenon as imitationleanring: distribution shift
        - Model errors push rollouts into slightly wrong states → those states are less trained → bigger errors → divergence from reality grows with horizon ($O(\epsilon T^2 )$)
        - Also v2.5 change the plicy to be good under the model, it makes the shift even larger
    5. Short model rollouts + real state branching (v3.0)
    ![Figure3](img/v3.0.png)
        - Collect some long real trajectoryies occasionally.
        - Sample states from anywhere along them including late time steps
        - From the each samples real state, run very short model rollouts
        - Train using both real + model data, typically with off-policy RL like Q-learning/off-policy actor-critic. Because the state distribution becomes a mixture distribution
- Terminology Map <!--(용어 등치/정의)-->
    - Policy gradient (Likelihood-ratio gradient estimator): Use sampling which doesn't contain a product of dynamics Jacobians
    - Pathwise gradient = backpropagation gradient: It contains long Jacobian products
- Why it matters <!--(왜 중요한가 1~2줄) 이걸 놓치면 다음 파트가 왜 나오는지 이해가 안 됨-->
    - This part explain why backprop through the leared model is unstable, why just use PG in the model still fails with long rollouts, and why modern model-based method often rely on short model rollouts + off-policy learning to control compounding model error
    - 
## Part 3
- Limitation → Fix<!--이전 파트 한계 → 이번 파트 해결책-->
    - Limitation: Attempting to rollout the model for long horizons causes errors to accumulate $O(\epsilon T^2 )$ and it diverges
    - Fix: Use model as a data generator instead of backprop
- Flow of this part <!--(어떤 흐름으로 강의가 진행되는지, 강의의 흐름을 반영하는건 맞지만 어떻게 알고리즘을 발전해나가는지 정리)-->
    1. Use off-policy backbone: use Q-leanring and also actor-critic variants work similiarly
    2. Classic Dyna
    ![Figure4](img/classic_dyna.png)
        - Use only 1-step model rollouts. If the model is accurate, performance will work well even thought really short estimation
        - 
    3. Dyna-style (modern MBRL v3.0 v)
    ![Figure5](img/dyna_style.png)
        - It only requires short as few as one step rollouts from model
        - still sees diverse states
    4. Model-accelerated off-policy RL
    ![Figure6](img/system_view.png)
    - General version of Dyna-style
        - Pros: Sample effecient because they're using their samples to train this model being used to amplify the data set. So training speed is fast and put more data to Q-learning process than collecting from MDP
        - Cons: If the model is not perfect, there's model bias.
- Terminology Map <!--(용어 등치/정의)-->
    - Model bias: errors in $\hat p / \hat r$ cause the learner to optimize the wrong objective.
    - MBA (Model-Based Acceleration): 모델 기반 데이터를 활용해 학습을 가속화
    - MVE (Model-Based Value Expansion): 모델 롤아웃을 통해 타겟 밸류(y)를 더 멀리까지 보고 추정하여 Q-러닝의 정확도를 높임
    - MBPO (Model-Based Policy Optimization): 장표의 6단계 절차와 가장 유사하며, 짧은 롤아웃을 통해 정책을 최적화. 요즘 표준으로 쓰이는 알고리즘
## Part 4
- Limitation → Fix<!--이전 파트 한계 → 이번 파트 해결책-->
    - Prev: Previous models focus on predict next state $s_{t+1}$. But if the state space is big, estimatevalue with learning physics can be inefficient
    - Fix: Predict which states will I visit instead of physics. Seperate value funtion ($V$) to reward $r$ and future visit frequency ($\mu$). So you don't need to retrain the model when the reward changes
- Professor’s Emphasis <!--(핵심 한 줄): 강의에서 교수님이 ‘딱 박아준’ 문장-->
    - The model's real job is to evaluate a policy
- Flow of this part <!--(어떤 흐름으로 강의가 진행되는지, 강의의 흐름을 반영하는건 맞지만 어떻게 알고리즘을 발전해나가는지 정리)-->
    1. Model is the method to calculate expected reward of policy not the physics engine.
    2. Rewrite the value function to future state distribution ($\mu$) and rewrad ($R$)
    3. Successor Representation ($\mu$): The discounted sum of all future states visited under the current policy
    4. Extension to Successor Features ($\psi$): To handle very large state spaces, $\mu$ is projected onto a low dimensional basis $\phi$
    5. C-learning: On the continuous state space, change to binary classification problem to learn SR
- Terminology Map <!--(용어 등치/정의)-->
    - 특징량 ($\phi$)
        - 상태 ($s$)가 가진 보잡한 정보를 우리가 다루기 쉬운 수치적 형태로 요약한 상태의 대표 속성
        - Atari 게임같은 Raw state는 데이터가 너무 방대하고 복잡하기 때문이 특징량을 상태 s를 입력받아 특정 숫자 (벡터)로 내뱉는 함수 $\phi(s)$임
        - 의사결정에 중요한 정보만 추출하여 차원을 축소함
        - $\phi(s)$가 현재 상태에서 얻는 즉각적인 특징이라면, $\psi(s)$는 미래에 얻게 될 특징량($\phi$)들의 총합을 나타냄
            - 특징량 ($\phi(s)$): 지금 사과가 있는지?
            - 계승 특성 ($\psi(s)$): 이 정책대로 진행하면 앞으로 사고를 몇개나 더 보게 될 것인지?

    - Successor Representation (SR, $\mu^\pi$)
        - 정의: 현재 상태 $s_t$ 에서 시작하여 정책 $\pi$ 를 따를 때 미래의 모든 시점에서 특정 상태 $s$ 에 도달할 할인된 누적 확률 (Discounted future occupancy)
        - SR은 오직 환경의 transition과 polit ($\pi$)에만 의존하며 보상($R$)과는 완전히 독립적
    - Successor Features (SF, $\phi^\pi$)
        - 상태 공간이 너무 크니 상태 자체가 아닌 특징의 방문 빈도를 계산하자
        - Atari 게임의 이미지처럼 상태 공간이 엄청나게 크거나 연속적일 경우 모든 상태에 대한 SR 벡터를 만드는것은 불가능
        - SR을 우리가 정의한 낮은 차원의 특징량 기저인 $\phi$에 투영한 것을 계승 특성 ($\psi$)라고 함
        - 만약 보상함수 $R(s)$가 특징량들의 선형 결합 ($\phi(s)^\top w$)으로 표현될 수 있으면, 전체 가치함수 $V(s)$ 또한 계승 특성의 선형 결합 ($\psi(s)^\top w$)로 즉시 계산이 가능함
        - $V^\pi(s_t)=\psi^\pi(s_t)^\top w$
- Why it matters <!--(왜 중요한가 1~2줄) 이걸 놓치면 다음 파트가 왜 나오는지 이해가 안 됨-->
    - Fast reward transfer
        - Without having to relearn the laws of physics or the entire policy, simply recalculating the reawrd function yields a new value function

## My Confusion & Clarification
- whole flow of this lecture
    1. MPC
        - 매 순간 지금부터 어떻게 할지 계왹을 짜고 시작함.
        - limitation
            - 계획을 짤 때 나중에 상황을 보고 바꿀 수 있다는 유연성을 계산에 넣지 못함. 즉 미래에 정보를 얻게 될 상황을 계획에 반영하지 못하고 무조건 안전한 길만 선택하려고 함. 시험지를 본 후 답을 고를 수 있는데 현재 상태에서는 시험지를 모르지만 답을 골라야하기 때문에 시험을 보지 않는 선택을 해버림
    2. Backprop through Model
        - 모델도 신경망이고 정책도 신경망니니까 체인룰로 backprop해서 보상을 높이자
        - limitation
            - 1단계 할 때마다 야코비안을 곱해야하기때문에 값이 터짐
        - 모델을 미분하는건 너무 어려우니까 미분을 안쓰는 policy gradient를 모델안에서 쓰자
    3. Sampling in Model
        - 모델을 직접 미분하지 않고 모델 안에서 sampling을 많이 돌린다음 그 결과로 학습
        - limitation
            - 시뮬레이션만 돌릴겨우 초반의 작은 실수가 눈덩이처럼 불어나서 현실이랑 괴리감이 생김
            - 오차 누적속도: 시간의 제곱 ($O(\epsilon T^2)$)
        - 모델 안에서 long horizental하게 돌리지 말자. 짧게 생각하자
    4. Dyna-style/MBPO
        - 실제 경험 (Real moemory)에서 아무 시점이나 하나 뽑기
        - 랜덤하게 뽑은 시점부터 1~10 스템까지만 모델로 시뮬레이션돌림
        - 이렇게 하면 오차가 커지기 전에 멈춰서 데이터의 정확도를 지키면서 실제 환경보다 훨씬 많은 데이터를 얻을 수 있음
        - 이 방법은 과거 policy와 모델의 상상이 섞여서 분포가 망가짐. 이걸 잘 처리하기 위해 Q-leanring같은 알고리즘을 사용함
    5. Successor Representation (SR)
        - 모델의 진짜 역할은 물리 법칙을 배우는 것이 아니라 정책을 평가하는게 진짜 역할
        - 미래에 어던 상태를 방문하게 될 것인지를 미래 점유 분포를 직접 예측
        - 특징
            - SR ($\mu^\pi$): 정책에는 의존하지만 보상 ($R$)과는 독립적. 따라서 보상이 바뀌어도 모델을 새로 학습할 필요가 없음 (Fast transfer)
            - Successor Features: 상태 공간이 너무 클 때, 상태 자체 대신 특징량 ($\phi$)의 방문 빈도를 계산하여 효율성을 높임



