# Terminology summary

1. Value iteration
    - 모델 $p(s'|s,a)$, 보상 $r(s,a)$를 안다고 할 때, Bellman optimality 연산자를 반복해서 최적 가치/정책을 구하는 동적계획법(DP)
    - Update equation
        - $Q(s,a)\leftarrow r(s,a)+\gamma \mathbb E_{s'}[V(s')],\quad V(s)\leftarrow \max_a Q(s,a)$
    - model-based + planning (다음 상태 분포를 계산)
2. Q-learning
    - 환경 모델을 몰라도 샘플 $(s,a,s',r)$로 최적 Q함수 $Q^*$를 직접 학습하는 방법(대표적 value-based RL)
    - Update equation
        - $Q(s,a)\leftarrow Q(s,a)+\alpha\big(r+\gamma \max_{a'}Q(s',a')-Q(s,a)\big)$
    - 다음 상태에서 가장 큰 Q를 쓰면 최적으로 감
    - off-policy, model 없이 value iteration을 샘플로 근사한 느낌
3. Policy gradient
    - 가치함수 대신 정책 파리미터 $\pi_\theta(a\mid s)$ 자체를 목표 $J(\theta)$를 최대화하도록 미분해서 직접 업데이트
    - Update
        - $\nabla_\theta J(\theta)=\mathbb E[\nabla_\theta \log \pi_\theta(a|s)\cdot \text{(advantage/return)}]$
    - 좋은 행동을 더 자주 뽑도록 확률을 올림
    - on-policy, 연속 제어에서 정책을 직접 다루기 쉬움
4. Actor-critic
    - Actor (policy) + Critic (V or Q)를 같이 학습하는 구조. Actor는 행동 확률을 업데이트 하고, Critic은 그 행동이 좋은지 (가치)를 추정해 저분산 학습 신호 (Advantage/TD error)를 알려줌
    - 장점
        - pure policy gradient: 분산 $\uparrow$
        - pure value-based: 연속 action에서 $max_a$가 어려움
        - actor-critic: 두개의 장점을 섞음
    - e.g. SAC
        - Critic: Update Q
        - Actor: Q를 많이 주는 행동을 뽑도록 (plus entropy)
5. Planning
    - 현재 상태에서 미래를 모델을 사용하여 좋은 행동/시퀀스를 계산으로 선택하는것
    - 예시: MPC, treesearch, DP 등등
    - model을 알고있거나 learned model이 이미 존재하는 등 모델이 필요함
6. On-policy
    - 학습에 쓰이는 데이터가 현재 학습중인 정책 $\pi$에서 나온 데이터
    - 장점: 이론이 단순하고 보통 안정적임
    - 샘플 효율 낮음 (정책이 바뀔 때마다 새 데이터가 필요함)
7. Off-policy
    - target policy와 data generation policy가 달라도 학습 가능
    - Q-learning, DQN, DDPG, SAC
    - 구현할 때 필요한거
        - replay buffer
        - importance sampling (optional)
    - 장점: 샘플 효율 좋음 (과거 데이터 재사용 가능)
    - 단점: 안정화 트릭/편향 이슈 (특히 function approximation)
8. Online RL
    - 학습 중에 환경과 계속 상호작용하면서 새 데이터를 받아 업데이트하는 설정
    - on-policy, off-policy 둘 다 online 가능
        - SAC는 off-policy지만 online으로 돌릴 수 있음 (계속 rollout+buffer)
9. Offline RL
    - 고정된 데이터셋만 가지고 학습. 학습 중 환경과 추가적인 상호작용 없음
    - 핵심 문제점: 데이터 밖으로 (action distribution 밖으로) 나가면 Q가 과대평가되기 쉬움 (OOD problem)
