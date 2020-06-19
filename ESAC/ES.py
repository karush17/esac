import os
import numpy as np
import sys, copy
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import optim
from model import NeuralNetwork
import pickle as pkl
import scipy.stats as ss
import gym
import random

checkpoint_name = './Checkpoint/'
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def crossover(args, STATE_DIM, ACTION_DIM, gene1, gene2):
        actor_1 = NeuralNetwork(STATE_DIM, ACTION_DIM.shape[0], args.hidden_size)
        actor_1.load_state_dict(gene1)
        actor_2 = NeuralNetwork(STATE_DIM, ACTION_DIM.shape[0], args.hidden_size)
        actor_2.load_state_dict(gene2)
        for param1, param2 in zip(actor_1.parameters(), actor_2.parameters()):

            # References to the variable tensors
            W1 = param1.data
            W2 = param2.data

            if len(W1.shape) == 2: #Weights no bias
                num_variables = W1.shape[0]
                # Crossover opertation [Indexed by row]
                num_cross_overs = random.randrange(num_variables*2)  # Lower bounded on full swaps
                for i in range(num_cross_overs):
                    receiver_choice = random.random()  # Choose which gene to receive the perturbation
                    if receiver_choice < 0.5:
                        ind_cr = random.randrange(W1.shape[0])  #
                        W1[ind_cr, :] = W2[ind_cr, :]
                    else:
                        ind_cr = random.randrange(W1.shape[0])  #
                        W2[ind_cr, :] = W1[ind_cr, :]

            elif len(W1.shape) == 1: #Bias
                num_variables = W1.shape[0]
                # Crossover opertation [Indexed by row]
                num_cross_overs = random.randrange(num_variables)  # Lower bounded on full swaps
                for i in range(num_cross_overs):
                    receiver_choice = random.random()  # Choose which gene to receive the perturbation
                    if receiver_choice < 0.5:
                        ind_cr = random.randrange(W1.shape[0])  #
                        W1[ind_cr] = W2[ind_cr]
                    else:
                        ind_cr = random.randrange(W1.shape[0])  #
                        W2[ind_cr] = W1[ind_cr]


def sample_noise(neural_net):
    # Sample noise for each parameter of the neural net
    nn_noise = []
    for n in neural_net.parameters():
        noise = np.random.normal(size=n.cpu().data.numpy().shape)
        nn_noise.append(noise)
    return np.array(nn_noise)

def evaluate_neuralnet(nn, env):
    # Evaluate an agent running it in the environment and computing the total reward
    obs = env.reset()
    game_reward = 0
    reward = 0
    done = False

    while True:
        # Output of the neural net
        obs = torch.FloatTensor(obs)
        action = nn(obs)
        action = np.clip(action.data.cpu().numpy().squeeze(), -1, 1)
        # action = action.data.numpy().argmax()
        # action = np.asarray([action])
        new_obs, reward, done, _ = env.step(action)
        
        obs = new_obs

        game_reward += reward

        if done:
            break
        
    return game_reward

def evaluate_noisy_net(STD_NOISE, noise, neural_net, env, elite_queue):
    # Evaluate a noisy agent by adding the noise to the plain agent
    old_dict = neural_net.state_dict()

    # add the noise to each parameter of the NN
    for n, p in zip(noise, neural_net.parameters()):
        p.data += torch.FloatTensor(n * STD_NOISE)

    elite_queue.put(neural_net.state_dict())

    # evaluate the agent with the noise
    reward = evaluate_neuralnet(neural_net, env)
    # load the previous paramater (the ones without the noise)
    neural_net.load_state_dict(old_dict)
    
    return reward

def worker(args, STD_NOISE, STATE_DIM, ACTION_DIM, params_queue, output_queue, elite_queue):
    # Function execute by each worker: get the agent' NN, sample noise and evaluate the agent adding the noise. Then return the seed and the rewards to the central unit
   
    env = gym.make(args.env)
    # env = CyclicMDP()
    actor = NeuralNetwork(STATE_DIM, ACTION_DIM.shape[0], args.hidden_size)
    while True:
        # get the new actor's params
        act_params = params_queue.get()
        if act_params != None:
            # load the actor params
            actor.load_state_dict(act_params)

            # get a random seed
            seed = np.random.randint(1e6)
            # set the new seed
            np.random.seed(seed)

            noise = sample_noise(actor)

            pos_rew = evaluate_noisy_net(STD_NOISE, noise, actor, env, elite_queue)
            # Mirrored sampling
            neg_rew = evaluate_noisy_net(STD_NOISE, -noise, actor, env, elite_queue)

            output_queue.put([[pos_rew, neg_rew], seed])
        else:
            break

def normalized_rank(rewards):
    '''
    Rank the rewards and normalize them.
    '''
    ranked = ss.rankdata(rewards)
    norm = (ranked - 1) / (len(ranked) - 1)
    norm -= 0.5
    return norm

def save_results(env, seed, time_list, max_rewards, min_rewards, avg_rewards, total_start_time, updates, noise_mut):
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


def eval(env, agent):
    avg_reward = 0.
    episodes = 10
    for _  in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state) #.select_action #torch.Tensor(state)
            # action = action.detach().cpu().numpy()

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            state = next_state
        avg_reward += episode_reward
    avg_reward /= episodes


    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
    print("----------------------------------------")


