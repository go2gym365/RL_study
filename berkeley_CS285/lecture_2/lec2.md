

 ## 0. Meta
- Course: CS285 RL
- Date: 25.12.07
- Lecturer: Lecture 2
- Source/Link:  https://www.youtube.com/watch?v=awfrsjYnJmw&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=8&pp=iAQB

---

## 1. Big Picture (오늘 강의의 한 줄 요약)
- you cannot say this model do imitation well even thought you dicrease "per-step supervised loss" well.
- higher quility of data make imitation learning work worse!

---

## 2. Key Concepts
<!-- 중요한 용어 / 개념 리스트업 (정확한 정의가 아니어도 됨, 나중에 수정 가능) -->
$s_t$ = state
$a_t$ = action
$O$ = observation: collected from real world. It's not same as state in MDP situation

-  The distributional shift problem
    - Observation collected from so expectation of P_data(Ot) is that If there's no under or over fitting, we would expect that log distribution under the distribution of $P_data(O_t)$. So good action has a high probablity .
    - but probability of good action would selected by $P_{\pi_\theta}$ not $P_{\text{data}}$. and those r not same. -> distributional shift!!
  
---

## 3. Important Equations / Diagrams
- Worst-case bound on total expected cost over horizon $T$:

$$
\mathbb{E}\Big[\sum_{t=1}^T c(s_t, a_t)\Big] = O(\epsilon T^2)
$$

(= per-step error는 $\epsilon$ 으로 작아도, 전체 에피소드 에러는 $T^2$까지 커질 수 있다.)

---

## 4. Main Logic / Algorithm Steps
DAgger
1. Train a policy ($\pi_\theta$) from human (expert) data
2. Run $\pi_\theta$ in the real circumstance to get dataset $D_\pi$
3. Get a human evaluation to label $D_\pi$ with actions $a_t$
4. Aggregate labeled data to original dataset D
5. Repeat 1~4 steps

---

## 5. Examples from the Lecture
<!-- 강의에서 든 예시, 직관, 비유, 데모 정리 -->
![Figure 1](img/figure1.png)
- Thogh per-step error is small as $\epsilon$, expected error amount can increase untill O($\epsilon T^2$)
  - probability of incorrect 1st step $\leq$ $\epsilon$: maximun cost: T
  - probability of incorrect 2nd step $\leq$ $(1-\epsilon)\epsilon$: maximun cost: T-1
  - probability of correct untill t-1 and incorrect at step t $\leq$ $(1-\epsilon)^{t-1}\epsilon$: maximun cost: T-(t-1)
- So you cannot say this model do imitation well even thought you dicrease "per-step supervised loss" well

---

## 6. My Confusions & Clarifications
<!-- 강의 들을 때 헷갈린 것들 + 나중에 찾아보고 이해한 내용 -->
### 6.1 What I didn’t understand (at first)
1. lots of researchers and students of RL are usually confusing State and Observation.

2. Behavioral cloning

3. How the error probability will increase untill $T^2$

### 6.2 What I found later (검색 / GPT / 책 보고 정리)
1. lots of researchers and students of RL are usually confusing State and Observation.
- In the theory or textbook, they assume a cercumstance of MDP. but in the research or experiment, we usually get a information of camera image, Lidar value .etc so those are observation because they aren't include all the information. They're actually POMDP. Also lot of code miscalled observation as a state. 
- So we should not confuse about the concept between State and Observation

2. Behavioral cloning
 -> Behavioral cloning is the moethod to fit 전문가 행동 using state by supervised-learning. But in the RL, Data is not IID so if there's a tiny ploblems, it will become compounding error and explode (the probability to make a big ploblem will be increase!)
- so collect a data smartly!
    - use very powerful model
    - make little wrng data to collect daggerish data 
    - data augmentation
    - use multitask learning

3. higher quility of data make imitation learning work worse! Is it same as like over fitting??
- Different! Bc High-qulity data from Levine is that the driver drive too perfectly so he always pass theh ideal way. So if the driver suddenly go through the wrong way(unfamilier state), the driver doesn't know what to do bc he has never experienced the same state.
- But it would be more firm and strong model, if the driver already experienced before and know how to restore it
- So the high-quility data makes distribution too narrow!
- Plus, algorithm like DAgger give a change to mistake and ask how to manage this situation. So policy can learn how to manage mistake situation.
