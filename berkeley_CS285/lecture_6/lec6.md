## 0. Meta
- Course: CS 285, Reinforcement Learning
- Date: 2025.12.
- Lecturer: Sergey Levine
- Source/Link: https://www.youtube.com/watch?v=wr00ef_TY6Q&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=21

---

## 1. Big Picture (one-sentence summary of this lecture)
<!-- 오늘 강의의 핵심 메시지/주제 한두 줄 -->
- 

---

## 2. Key Concepts
<!-- 중요한 용어 / 개념 리스트업 (정확한 정의가 아니어도 됨, 나중에 수정 가능) -->
- **$\hat{Q_{i,t}}_t$**: It estimates of expected reward if we take action $a_{i,t}$ in state $s_{i,t}$
    - It made by a single rollout so it would have big variance



---

## 3. Important Equations / Diagrams
<!-- 수식, 그림/도식 설명. 수식은 LaTeX로 적어두면 나중에 재사용하기 좋음 -->
- Value function
    - $V^\pi(s_t)$: The total expected reward from $s_t$ when you following policy $\pi$ thereafter
    - $Q^\pi(s_t,a_t)$: The total xpected reward from taking $a_t$ in $s_t$ when you following policy $\pi$ thereafter
    - $A^\pi(s_t,a_t)$: How much better $a_t$ is as compared to the average performance of your policy pi in state $s_t$

---

## 4. Main Logic / Algorithm Steps
<!-- 강의에서 설명한 절차, 알고리즘 흐름을 단계별로 정리 -->
1. REINFORCE uses a single sample estimate from a single rollout to approximate a very complex expectation, so because the policy and the MDP have randomness, the estimator has very high variance
2. In policy gradient methods, we can subtract a baseline $b(s_t)$ and the expected policy gradient does not change. The baseline can depend on the state and it's safe, but if it depends on the action it can lead to bias.
3. Fitting a Q function or and A function is harder because they depend on both the state and the action, so you need more samples. However $V^\pi(s)$ depends on the only state so it's more convenient to fit and more sample efficient. So lets fit $V^\pi(s)$ first and then derive the advantage approximately
4. Two common target to fit $V^\pi(s)$
    - MC evaluation
        - pros: Unbiased
        - cons: High variance because it uses the sum of rewards along the remainder of the trajectory. (especially for long trajectory)
    - TD evaluation
        - pros: Lower variance. Because it uses the current reward plus the value of the next state
        - cons: It can have higher biase if $\hat V^\pi$ Is incorrect

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
- 

### 6.2 What I found later (from web search, GPT, and books)
- 

---

## 7. Connections
<!-- 이 강의 내용이랑 연결되는 것들: 이전 강의, 다른 과목, 내 연구/프로젝트 등 -->
- Relation to previous lectures:
- Relation to my projects/research:
- Real-world / paper connections:

---

## 8. Keywords to Search Later
<!-- 더 깊게 보고 싶은 키워드, 논문 키워드, 용어 -->
- 
- 

---

## 9. Action Items / TODO
<!-- 다음에 할 것들 체크리스트 -->
- [ ] 슬라이드 다시 보기 (시간: )
- [ ] 과제/코드에 이 개념 적용해보기
- [ ] 관련 논문/블로그 하나 찾아 읽기: 