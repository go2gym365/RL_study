from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        # state를 Q-network에 넣기 위해 텐서변환이랑 차원 붙이기
        observation = ptu.from_numpy(np.asarray(observation))[None]
        # print(observation.shape)
        # print(observation)
        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        if np.random.random() < epsilon:
            # torch.randint 정수 난수 텐서 생성
            action = torch.randint(0, self.num_actions, size=(1, ), device=observation.device)
        else:
            with torch.no_grad():
                q_values = self.critic(observation) # 현재 상태에서 각 액션별 q값 계산
                action = torch.argmax(q_values, dim=1)

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(student): compute target values
            next_q_online = self.critic(next_obs) # online 네트워크로 Q 계산
            next_q_target = self.target_critic(next_obs) # target 네트워크로 Q 계산

            if self.use_double_q:
                next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
                next_q_values = torch.gather(next_q_target, 1, next_actions).squeeze(1)
            else:
                next_q_values = torch.max(next_q_target, dim=1).values
            
            # Bellman target 계산
            target_values = reward + self.discount * (1.0 - done.float()) * next_q_values

        # TODO(student): train the critic with the target values
        qa_values = self.critic(obs)
        # print(qa_values.shape)
        # print(action.shape)
        if action.dim() == 1:
            action = action.unsqueeze(1)
        # torch.gather(input, dim, index) : input 텐서에서 index 위치의 값을 dim 방향으로 가져옴
        # out[i, j] = qa[i, index[i, j]]
        # 각 배치 샘플마다 실제로 한 action에 해당하는 Q값을 선택
        q_values = torch.gather(qa_values, 1, action).squeeze(1) 
        loss = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): update the critic, and the target if needed
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)

        if step % self.target_update_period == 0:
            self.update_target_critic()

        return critic_stats
