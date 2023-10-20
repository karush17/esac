"""Implements the Soft Actor Critic algorithm."""

from typing import Any, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC(object):
    """SAC learner for RL tranining.
    
    Attributes:
        gamma: discount factor.
        tau: target update constant.
        alpha: temperature parameter for entropy.
        policy_type: neural network type.
        target_update_interval: steps before target network updates.
        automatic_entropy_tuning: whether to tune entropy parameter.
        device: cuda/cpu device.
        critic: Q function critic network.
        critic_optim: critic optimiser.
        critic_target: Q function critic target network.
        target_entropy: entropy target value.
        log_alpha: log of entropy parameter.
        alpah_optim: entropy parameter optimiser.
        policy: agent policy network.
        policy_optim: policy network optimiser.
    """
    def __init__(self, num_inputs: int, action_space: int, args: Any) -> None:

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.critic = QNetwork(num_inputs, action_space.shape[0],
                               args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0],
                                      args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True,
                                             device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs,
                                         action_space.shape[0],
                                         args.hidden_size,
                                         action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs,
                                              action_space.shape[0],
                                              args.hidden_size,
                                              action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state: np.ndarray,
                      evaluate: bool = False) -> np.ndarray:
        """Select an action from the policy."""
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory: Any,
                          batch_size: int,
                          updates: Any) -> Tuple[float, float,
                                                 float, float, float]:
        """Update parameters of the network."""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch,
                                                                  next_state_action)
            min_qf_next_target = torch.min(qf1_next_target,
                                           qf2_next_target)- self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        critic_loss = min(qf1_loss,qf2_loss)

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi
                                             + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.Tensor(0.).to(self.device)
            alpha_tlogs = torch.Tensor(self.alpha)


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return critic_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

