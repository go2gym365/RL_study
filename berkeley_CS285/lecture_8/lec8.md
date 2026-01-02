## 0. Meta
- Course: CS 285, Reinforcement Learning
- Date: 2025.01.02
- Lecturer: Sergey Levine
- Source/Link: https://www.youtube.com/watch?v=7-D8RL3D6CI&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=30

---

## 1. Big Picture (one-sentence summary of this lecture)
<!-- 오늘 강의의 핵심 메시지/주제 한두 줄 -->
- 

---

## 2. Key Concepts
<!-- 중요한 용어 / 개념 리스트업 (정확한 정의가 아니어도 됨, 나중에 수정 가능) -->
- **IID (Independent and Identically Distributed):**
    - Each data sample is Independent/ Identically Distributed each other
    - At the Online Q-learning, $s_t, s_{t+1}, ...$ are continous state and connected each other. So they are not Independent and distribution is keep changing. That's why IID asumption is destory on online Q-learning
- **SGD (Stochastic Gradient Descent)**
    - Instead of computing the gradient on the entire data set, we use a method to approximate and update it using a subset (or even just one) of the sample
    - 
- **Replay buffer** 
![Figure1](img/qlearning_replaybuffer.png)
    - If the function approximator had seen all transitions at once, it might have fit them well. But instead, it only ever sees a small, highly correlated window, which gives it just enough time to overfit and not enough context to generalize
    - steps
        1. collect dataset $(s, a, s', r)$ using some policy, add it to $(B)$
        2. Sample a batch $(s, a, s', r)$ from $B$
        3. Compute target value for each transition and Q-function update by batch per batch so that we can get lower variance
        - We can repeat step2, 3 K times but if the K is larger it will be more efficient. Or If the K is 1 and collect many datas that's the classic Deep Q-learning

- **Polyak averaging**
    - uneven lag problem: When the target network is copied all at once at fixed intervals, he amount of lag vsries significantly. Right after an update, the lag is very small, while just before the next update, the lag can become vary large, which may make training unstable.
    - Soft update (Polyak Averaging): To solve htis issue, Polyak averaging updates the target network slightly and smoothly at every step
    - equation: $\phi' \leftarrow \tau \phi' + (1 - \tau)\phi$
    - In this equation we usualy use large value of $\tau$ like 0.999 to make target network update current network really slowly and stable
- **Term 3** – 

---

## 3. Important Equations / Diagrams
<!-- 수식, 그림/도식 설명. 수식은 LaTeX로 적어두면 나중에 재사용하기 좋음 -->
- Equation:
  - 
- Notes:
  - 

---

## 4. Main Logic / Algorithm Steps
<!-- 강의에서 설명한 절차, 알고리즘 흐름을 단계별로 정리 -->
- **Q-learning**
![Figure2](img/DQN.png)
    - Q-learning with Repaly buffer + Traget Network
        1. Save target network parameters: Copy the current network parameters $\phi$ to the target network $\phi'$. so that $\phi' \leftarrow \phi$
        2. Collect Dataset: Use a policy to collect $(s_i, a_i, s'_i, r_i)$ transition paris and add them to the replay buffer $B$
        3. Sample a batch: Randomly sample a batch of data from the $B$ to be used for training
        4. Update Parameters: Update the parameters of the current Network $Q_\phi$ based on target values calculated using the target network $Q_{\phi'}$
        Loop structure: Step 3 and 4 are repeated K times. After the inner loop completes, the algorithm returns to Step 2 to collect more data. This entire process is performed N times before the target network is updated again in Step 1
        - If you set the K as 1, It's totally same as "Classic Deep Q-learning"
    - Classic Q-learning (DQN)
        1. Take Action and observe: Excute and action $a_i$ in the environment, observe the resulting transition $(s_i, a_i, s'_i, r_i)$ and add it to the buffer $B$
        2. Sample Mini-batch: Uniformly sample a mini-batch $(s_j, a_j, s'_j, r_j)$ from the $B$. This random sampling helps to break the correlation between sequential samples.
        3. Compute target: Calculate the target value, which serves as the ground truth for the regression, using the target network
        4. Update current network: Update the current network paremetes $\phi$ to minimize the error between $Q_\phi$ and the previously caculated target $y_j$
        5. Copy target Network: Every N steps, copy the current parameters $\phi$ to the target network $\phi'$. This stabilizes training by resolbing the moving target proble

        

---

## 5. Examples from the Lecture
<!-- 강의에서 든 예시, 직관, 비유, 데모 정리 -->
- Example 1:
- Example 2:
- Intuition: 

---

## 6. My Confusions & Clarifications
<!-- 강의 들을 때 헷갈린 것들 + 나중에 찾아보고 이해한 내용 -->
### 6.1 What I didn’t understand (at first)
- Q-learning에서 업데이트 식이 왜 Gradient descent처럼 보이지만 진짜 gradient descent가 아닌지

### 6.2 What I found later
- Q-learning에서 업데이트 식이 왜 Gradient descent처럼 보이지만 진짜 gradient descent가 아닌지
    - 업데이트 식 $y_i = r_i + \gamma \max_{a'} Q_\phi(s', a')$을 보면 $Q_\phi$가 들어있음. 하지만 실제 업데이트에서는 traget 쪽의 Q에는 gradient를 계산을 안하기 때문에 체인룰을 깨버린다. 

---