"""Stores utilities for evolution."""

import math
import torch

def create_log_gaussian(mean: torch.Tensor,
                        log_std: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Creates a log gaussian distribution over the tensor.
    
    Args:
        mean: meanof the distribution.
        log_std: log standard deviation of the distribution.
        t: tensor.
    
    Returns:
        log_p: log probability of the tensor sample.
    """
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs: torch.Tensor, dim: int = None,
              keepdim: bool = False) -> torch.Tensor:
    """Implements the log-sum-exp function.
    
    Args:
        inputs: input tensor.
        dim: dimension to take logsumexp along.
        keepdim: keep auxilary dimension.
    
    Returns:
        outputs: logsumexp transformed tensor.
    """
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau) -> None:
    """Soft target update."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source) -> None:
    """Hard target update."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def dm_wrap(state, wrap=True) -> torch.Tensor:
    """Fetch observation from wrapped environment."""
    if wrap==True:
        state = state["observations"]
    return state
         
