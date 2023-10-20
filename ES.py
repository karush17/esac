"""Implements Evolution Strategies with Soft Winner Selection."""

from typing import Dict, Any, List

import time
import random

import pickle as pkl
import numpy as np
import torch
import scipy.stats as ss
import gym

from utils import dm_wrap
from model import NeuralNetwork

checkpoint_name = './Checkpoint/'
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')



def crossover(args: Dict[str, Any], STATE_DIM: int, ACTION_DIM: int,
              gene1: Any, gene2: Any) -> Any:
        """Implements genetic crossovers between parameters.
        
        Args:
            args: experiment arguments.
            STATE_DIM: state dimension.
            ACTION_DIM: action dimension.
            gene1: first parameter.
            gene2: second parameter.
        
        Returns:
            crossover_gene: parameter with crossovered values.
        """
        actor_1 = NeuralNetwork(STATE_DIM, ACTION_DIM.shape[0],
                                args.hidden_size)
        actor_1.load_state_dict(gene1)
        actor_2 = NeuralNetwork(STATE_DIM, ACTION_DIM.shape[0],
                                args.hidden_size)
        actor_2.load_state_dict(gene2)
        for param1, param2 in zip(actor_1.parameters(), actor_2.parameters()):

            W1 = param1.data
            W2 = param2.data

            if len(W1.shape) == 2:
                num_variables = W1.shape[0]
                num_cross_overs = random.randrange(num_variables*2)
                for i in range(num_cross_overs):
                    receiver_choice = random.random()
                    if receiver_choice < 0.5:
                        ind_cr = random.randrange(W1.shape[0])
                        W1[ind_cr, :] = W2[ind_cr, :]
                    else:
                        ind_cr = random.randrange(W1.shape[0])
                        W2[ind_cr, :] = W1[ind_cr, :]

            elif len(W1.shape) == 1:
                num_variables = W1.shape[0]
                num_cross_overs = random.randrange(num_variables)
                for i in range(num_cross_overs):
                    receiver_choice = random.random()
                    if receiver_choice < 0.5:
                        ind_cr = random.randrange(W1.shape[0])
                        W1[ind_cr] = W2[ind_cr]
                    else:
                        ind_cr = random.randrange(W1.shape[0])
                        W2[ind_cr] = W1[ind_cr]

            param1.data = W1
            param2.data = W2
        
        return actor_2.state_dict()


def sample_noise(neural_net: torch.Module) -> np.ndarray:
    """Samples noise for the neural network parameters.
    
    Args:
        neural_net: neural network object.
    
    Returns:
        noise: generated random noise.
    """
    nn_noise = []
    for n in neural_net.parameters():
        noise = np.random.normal(size=n.cpu().data.numpy().shape)
        nn_noise.append(noise)
    return np.array(nn_noise)


def evaluate_neuralnet(nn: torch.Module, env: Any, wrap: Any) -> float:
    """Evaluates the network in an environment.
    
    Args:
        nn: neural network.
        env: evaluation environment.
        wrap: environment wrapper.
    
    Returns:
        reward: averaged reward across evaluation episodes.
    """
    obs = dm_wrap(env.reset(), wrap=wrap)
    game_reward = 0
    reward = 0
    done = False

    while True:
        obs = torch.FloatTensor(obs)
        action = nn(obs)
        action = np.clip(action.data.cpu().numpy().squeeze(), -1, 1)
        new_obs, reward, done, _ = env.step(action)        
        obs = dm_wrap(new_obs, wrap=wrap)
        game_reward += reward

        if done:
            break
        
    return game_reward

def evaluate_noisy_net(STD_NOISE, noise,
                       neural_net, env, elite_queue, wrap) -> float:
    """Evaluates the noisy network.
    
    Args:
        STD_NOISE: standard deviation of noise distribution in parameters.
        noise: noise level.
        neural_net: neural net architecture.
        env: agent environment.
        elite_queue: queue of elite learners in population.
        wrap: environment wrapper.
    
    Returns:
        reward: averaged reward after eavluation.
    """
    old_dict = neural_net.state_dict()

    for n, p in zip(noise, neural_net.parameters()):
        p.data += torch.FloatTensor(n * STD_NOISE)

    elite_queue.put(neural_net.state_dict())

    reward = evaluate_neuralnet(neural_net, env, wrap)
    neural_net.load_state_dict(old_dict)    
    return reward

def worker(args, STD_NOISE, STATE_DIM, ACTION_DIM,
           params_queue, output_queue, elite_queue) -> None:
    """Evlautes the network and puts the rewards in queue.
    
    Args:
        args: experiment arguments.
        STD_NOISE: standard deviation of the noise distribution.
        STATE_DIM: dimension of the state.
        ACTION_DIM: dimension of the action.
        params_queue: queue of agent parameters.
        output_queue: queue of rewards.
        elite_queue: queue of elite learners.
    """
    if 'dm2gym' in args.env:
        env = gym.make(args.env, environment_kwargs={'flat_observation': True})
        wrap = True
    else:
        env = gym.make(args.env)
        wrap = False

    # env = CyclicMDP()
    actor = NeuralNetwork(STATE_DIM, ACTION_DIM.shape[0], args.hidden_size)
    while True:
        act_params = params_queue.get()
        if act_params != None:
            actor.load_state_dict(act_params)
            seed = np.random.randint(1e6)
            np.random.seed(seed)
            noise = sample_noise(actor)
            pos_rew = evaluate_noisy_net(STD_NOISE, noise,
                                         actor, env, elite_queue, wrap)
            neg_rew = evaluate_noisy_net(STD_NOISE, -noise,
                                         actor, env, elite_queue, wrap)
            output_queue.put([[pos_rew, neg_rew], seed])
        else:
            break

def normalized_rank(rewards: np.ndarray) -> np.ndarray:
    """Ranks the reward and normalizes them.
    
    Args:
        rewards: agent rewards.
    
    Returns:
        norm: normalized rewards.
    """
    ranked = ss.rankdata(rewards)
    norm = (ranked - 1) / (len(ranked) - 1)
    norm -= 0.5
    return norm

def save_results(env: Any, seed: int, time_list: List[int],
                 max_rewards: np.ndarray, min_rewards: np.ndarray,
                 avg_rewards: np.ndarray, total_start_time: int,
                 updates: Any, noise_mut: Any) -> None:
    """Saves results to a pickle dictionary.
    
    Args:
        env: environment.
        seed: random seed.
        time_list: list of timesteps.
        max_rewards: maximum rewards among learners.
        min_rewards: minimum reward among learners.
        avg_rewards: average rewards of learners.
        total_start_time: initial timestep.
        updates: gradient updates.
        noise_mut: noise mutation level.
    """
    data_save = {}
    data_save['time'] = time_list
    data_save['max_rewards'] = max_rewards
    data_save['min_reward'] = min_rewards
    data_save['avg_reward'] = avg_rewards
    data_save['total_time'] = total_start_time - time.time()
    data_save['SAC_updates'] = updates
    data_save['noise'] = noise_mut

    with open(checkpoint_name+'data_'+str(env)+'_'+str(seed)+'.pkl', 'wb') as f:
        pkl.dump(data_save, f)


def evaluate(env: Any, agent: torch.Module) -> None:
    """Evaluate the learned policy.
    
    Args:
        env: environment.
        agent: learned policy of the agent.
    """
    avg_reward = 0.
    episodes = 10
    for _  in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            state = next_state
        avg_reward += episode_reward
    avg_reward /= episodes


    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(episodes,
                                                      round(avg_reward, 2)))
    print("----------------------------------------")

