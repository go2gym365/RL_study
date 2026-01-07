"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    2. forward
    3. update
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))

    mlp = nn.Sequential(*layers)
    return mlp


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to continuous
    actions.

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        self.mean_net = build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.mean_net.to(ptu.device)
        self.logstd = nn.Parameter(

            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None, :]

        observation = ptu.from_numpy(observation.astype(np.float32))
        dist = self.forward(observation)

        # 보통 BC/DAgger에서는 mean을 쓰거나 sample을 씀
        action = dist.mean  # 또는 dist.sample()
        return ptu.to_numpy(action)

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        # TODO: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it. For example, you can return a torch.FloatTensor. You can also
        # return more flexible objects, such as a
        # `torch.distributions.Distribution` object. It's up to you!
        # 관측을 넣으면 정책을 반환하는 함수
        mean = self.mean_net(observation) # mean_net: multi layer perceptron
        scale_dist = torch.exp(self.logstd) # 표준편차의 로그를 학습하기 때문에 지수화 항상 양수 나옴 
        dist = distributions.Normal(mean, scale_dist) # 정규분포를 나타내는 클래스
        return dist # 확률분포 객체를 반환. 

    def update(self, observations, actions): # 관측 s가 들어오면 정책이 expert action을 잘 내도록 학습. maximum likelihood
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # TODO: update the policy and return the loss
        observations = ptu.from_numpy(observations) # numpy 배열을 pytorch 텐서로 변환 + GPU 사용시 to(device)도 자동으로 처리됨
        actions = ptu.from_numpy(actions)
        
        dist = self.forward(observations) # 관측 배치에 대한 정책 분포를 얻음

        loss = -dist.log_prob(actions).sum(dim=-1).mean() # 음의 로그 확률 밀도 함수 값을 계산하여 손실을 구함. sum(dim=-1): 각 샘플에 대한 로그 확률을 합산, mean(): 배치 전체에 대한 평균 손실 계산
                                                          # optimizer는 기본적으로 loss를 최소화 하려고 하기 때문에 음수 부호를 붙여줌으로써 최대우도 로직으로 변환
                                                          # sum(dim=-1) 연속 행동이 여러 차원일 때, 행동 전체의 확률을 계산하려면 각 차원의 log확률을 더해야함
        self.optimizer.zero_grad() # 이전 배치에서 쌓인 그라디언트 초기화
        loss.backward() # loss를 기준으로 mean_net과 logstd에 대한 gred 계산
        self.optimizer.step() 

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
