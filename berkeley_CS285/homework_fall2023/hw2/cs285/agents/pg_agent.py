from typing import Optional, Sequence
import numpy as np
import torch

from cs285.networks.policies import MLPPolicyPG
from cs285.networks.critics import ValueCritic
from cs285.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.

        # 시간축을 concat하기 때문에 padding같은건 필요없음. 즉 샘플단위 학습
        obs_flat = np.concatenate(obs)
        actions_flat = np.concatenate(actions)
        rewards_flat = np.concatenate(rewards)
        terminals_flat = np.concatenate(terminals)
        q_values_flat = np.concatenate(q_values)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs_flat, rewards_flat, q_values_flat, terminals_flat
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        info: dict = self.actor.update(obs_flat, actions_flat, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        # Baseline을 쓰는 경우 advantage 를 V 기반으로 계산하기 때문에 V(s)를 잘 맞추는 네트워크가 필요함
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            critic_info: dict = {}
            for _ in range(self.baseline_gradient_steps): # critic은 회귀이기 때문에 setps만큼 여러번 돌려서 정확도를 높임
                critic_info = self.critic.update(obs_flat, q_values_flat)

            info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO: use the helper function self._discounted_return to calculate the Q-values
            # q_values = []
            # for reward in rewards:
            #     item = np.array(self._discounted_return(reward))
            #     q_values.append(item)
            q_values = [np.array(self._discounted_return(reward)) for reward in rewards]

        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
            q_values = [np.array(self._discounted_reward_to_go(reward), dtype=np.float32) for reward in rewards]
        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # TODO: if no baseline, then what are the advantages?
            advantages = q_values.copy() # critic이 없을때 advantage는 q_value와 동일. A=Q
        else:
            # TODO: run the critic and use it as a baseline 
            values = ptu.to_numpy(self.critic(ptu.from_numpy(obs))).squeeze() # baseline 있으니까 v계산
            assert values.shape == q_values.shape 

            if self.gae_lambda is None: # GAE 안쓰고 baseline만 쓸때 Q-V로 advantage 계산하기. A=Q-V
                # TODO: if using a baseline, but not GAE, what are the advantages?
                advantages = q_values - values
            else: 
                # TODO: implement GAE
                batch_size = obs.shape[0] 

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1) # 힌트대로 더미 하나 추가해서 계산 편하게 하도록

                for i in reversed(range(batch_size)): # A_{t+1}이 있어샤 A_t를 계산할 수 있으니까 reverssed로 뒤부터 돌도록
                    nonterminal = 1.0 - terminals[i] # terminal state면 1, nonterminal이면 0. terminal이면 다음 상태가 없으니까 0 곱해져서 td error도 0이 됨. 그래서 각 에피소드간에 정보를 안섞이게    
                                                     # 뒤에서부터 계산하니 ep1의 termial이 t이면 ep2의 첫스텝은 t+1이니 여기서 안끊어주면 정보가 ep1로 섞여들어감
                    delta = rewards[i] + self.gamma * values[i+1] * nonterminal - values[i] # TD error 계산부분 
                    advantages[i] = delta + self.gamma * self.gae_lambda * nonterminal * advantages[i+1]
                    # gae_lamda가 0이면 1-step td, 1이면, 몬테카를로에 가까워짐. 더 긴 horizon 반영하는것 

                # remove dummy advantage
                advantages = advantages[:-1] #처음에 더해준 더미 빼주기

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            # 스케일이 배치마다 다를 수 있으니까 배치 안에서 평균 0, 표준편차 1로 정규화. 분모의 1e-8은 0나누기 방지용

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        ret = 0.0
        for t, r in enumerate(rewards):
            ret += (self.gamma ** t) * r
        return [ret for _ in rewards]


    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        return_to_go = np.zeros(len(rewards))
        running = 0.0
        for i in reversed(range(len(rewards))):
            running = rewards[i] + self.gamma * running
            return_to_go[i] = running
        return return_to_go.tolist()