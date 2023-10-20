"""Implements RL modules."""

from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m: Any):
    """Xavier Weight initialization."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    """Q Value Network.
    
    Attributes:
        linear1: layer 1.
        linear2: layer 2.
        linear3: layer 3.
    """
    def __init__(self, num_inputs: int, hidden_dim: int):
        """Initialize the network."""
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Pass obervation through the network."""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    """Q Value Network.
    
    Attributes:
        linear1: layer 1.
        linear2: layer 2.
        linear3: layer 3.
        linear4: layer 1 (second net).
        linear5: layer 2 (second net).
        linear6: layer 3 (second net).
    """

    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int):
        """Initialize the network."""
        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pass observations through the network."""
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1.clone())

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2.clone())

        return x1, x2


class GaussianPolicy(nn.Module):
    """Implements a Gaussian Policy.
    
    Attributes:
        linear1: layer 1.
        linear2: layer 2.
        mean_linear: mean layer.
        log_std_layer: log standard deviation layer.
        action_scale: scale of actions.
        action_bias: bias in actions.
    """
    def __init__(self, num_inputs: int, num_actions: int,
                 hidden_dim: int, action_space: int = None):
        """Initializes the network."""
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = torch.Tensor(1.)
            self.action_bias = torch.Tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Pass observations through the network."""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor,
                                                   torch.Tensor, torch.Tensor]:
        """Samples an action from the Gaussian distribution."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device: str):
        """Moves tensor to device."""
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    """Implements a Deterministic Policy.
    
    Attributes:
        linear1: layer 1.
        linear2: layer 2.
        mean: mean layer.
        noise: noise in the action.
        action_scale: scale of actions.
        action_bias: bias in actions.
    """

    def __init__(self, num_inputs: int, num_actions: int,
                 hidden_dim: int, action_space: int = None):
        """Initializes the policy object."""
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Pass observations through the network."""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action from the policy."""
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.Tensor(0.), mean

    def to(self, device):
        """Move tensor to device."""
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


class NeuralNetwork(nn.Module):
    """Implements a neural network object.
    
    Attributes:
        linear1: layer 1.
        linear2: layer 2.
        mean_linear: mean layer.
        log_std_linear: log standard deviation layer.
    """

    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int):
        """Initializes the network object."""
        super(NeuralNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Pass observations through the network."""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean
