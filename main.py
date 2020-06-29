import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse, sys
import datetime, copy
import gym
import numpy as np
import random
import itertools, time, math
import torch
import pickle as pkl
from sac import SAC
from model import NeuralNetwork
from parser import build_parser
from ES import crossover, sample_noise, evaluate_neuralnet, evaluate_noisy_net, worker, normalized_rank, save_results, eval
from replay_memory import ReplayMemory
from torch import optim
import torch.multiprocessing as mp
mp.set_start_method('spawn',force=True)

torch.autograd.set_detect_anomaly(True)
os.sched_setaffinity(os.getpid(), {0})
os.system("taskset -p 0xffffffffffffffffffffffff %d" % os.getpid())

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    checkpoint_name = './Checkpoint/'

    if not os.path.exists(checkpoint_name):
        os.makedirs(checkpoint_name)

                            ######################################### INITIALIZE SETUP #############################################

    # Parse Arguments
    parser = build_parser()
    args = parser.parse_args()

    # Environment
    env = gym.make(args.env)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space
    NUM_WINNERS = int(args.elite_rate*args.pop)
    BATCH_SIZE = args.pop
    STD_NOISE = args.mutation
    NUM_GRAD_MODELS = args.grad_models
    l1_loss = torch.nn.SmoothL1Loss()
    total_start_time = time.time()

    # Handle exceptions
    if NUM_GRAD_MODELS > BATCH_SIZE:
        print("Grad Models cannot be greater than Population size")
        quit()
    elif args.elite_rate > 1:
        print('Elite rate cannot be greater than 1')
        quit()

    # Initialize the ES mother paramters and optimizer
    actor = NeuralNetwork(STATE_DIM, ACTION_DIM.shape[0], args.hidden_size)
    optimizer = optim.Adam(actor.parameters(), lr=args.lr_es)
    torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.999, last_epoch=-1)

    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay = 1200000

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    # Worker Process Queues
    output_queue = mp.Queue(maxsize=args.pop)
    params_queue = mp.Queue(maxsize=args.pop)
    elite_queue = mp.Queue(maxsize=int(2*args.pop))

    # Agent
    agent = SAC(STATE_DIM, ACTION_DIM, args)
    sac_episodes = args.sac_episodes

    # Memory
    memory = ReplayMemory(args.replay_size)
    processes = []
    elite_list = []

    # Training Loop
    total_numsteps = 0
    updates = 0
    time_list = []
    max_rewards = []
    min_rewards = []
    avg_rewards = []
    noise_mut = []
    total_time = 0
    critic_loss = 0

    # Create and start the processes
    for _ in range(args.workers):
        p = mp.Process(target=worker, args=(args, STD_NOISE, STATE_DIM, ACTION_DIM, params_queue, output_queue, elite_queue))
        p.start()
        processes.append(p)

    # Initialize Elite list
    for _ in range(0,NUM_WINNERS):
        elite_list.append(actor.state_dict())


                            ######################################### TRAINING LOOP #############################################

    # Execute the main loop
    for n_iter in range(1,int(args.num_steps)):
        it_time = time.time()
        total_numsteps += env._max_episode_steps

        if total_numsteps > args.num_steps:
            break

        batch_noise = []
        batch_reward = []
        batch_loss = []
        batch_ratio = []
        
        # Crossover between Winners and Population Actors
        for h in range(0,int(NUM_WINNERS)):
            dict_copy = copy.deepcopy(actor.state_dict())
            crossover(args, STATE_DIM, ACTION_DIM, elite_list[h], dict_copy)
            params_queue.put(dict_copy)

        # Standard Population actors
        for h in range(NUM_WINNERS,BATCH_SIZE-NUM_GRAD_MODELS):
            params_queue.put(actor.state_dict())
        
        # Crossover between Grad models and Population Actors
        for _ in range(int(NUM_GRAD_MODELS)):
            # dict_copy = copy.deepcopy(actor.state_dict())
            agent_copy = copy.deepcopy(agent.policy.state_dict())
            # crossover(args, STATE_DIM, ACTION_DIM, agent_copy, dict_copy)
            params_queue.put(agent_copy)


                            ######################################### SAC UPDATE #############################################

        if total_numsteps % int(5*env._max_episode_steps) == 0 and random.random() < epsilon_by_frame(total_numsteps):

            # total_numsteps += int(sac_episodes*env._max_episode_steps)
            for i_episode in range(sac_episodes):
                episode_reward = 0
                episode_steps = 0
                done = False
                state = env.reset()

                while not done:
                    if args.start_steps > total_numsteps:
                        action = env.action_space.sample()
                    else:
                        action = agent.select_action(state)  # Sample action from policy

                    if len(memory) > args.batch_size:
                        # Number of updates per step in environment
                        for i in range(args.updates_per_step):
                            # Update parameters of all the networks
                            critic_loss, qf_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                            updates += 1

                    next_state, reward, done, _ = env.step(action) # Step
                    episode_reward += reward
                    episode_steps += 1

                    # Ignore the "done" signal if it comes from hitting the time horizon.
                    # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                    mask = 1 if episode_steps == env._max_episode_steps else float(not done)

                    memory.push(state, action, reward, next_state, mask) # Append transition to memory

                    state = next_state


                        ######################################### COLLECT DATA FROM QUEUE #############################################


        # receive from each worker the results (the seed and the rewards)
        
        for i in range(BATCH_SIZE):
            p_rews, p_seed = output_queue.get()
            
            np.random.seed(p_seed)
            noise = sample_noise(actor)
            batch_noise.append(noise)
            batch_noise.append(-noise)

            batch_reward.append(p_rews[0]) # reward of the positive noise
            batch_reward.append(p_rews[1]) # reward of the negative noise

        
        elite_array = [] # only a placeholder for getting parameters from queue
        elite_list = [] # stores weights of winners
        for _ in range(int(2*BATCH_SIZE)):
            elite_array.append(elite_queue.get())
        elite_list = [x for _,x in sorted(zip(batch_reward,elite_array),reverse=True)]


                            ######################################### ES UPDATE #############################################

        max_rewards.append(max(batch_reward))
        min_rewards.append(min(batch_reward))
        avg_rewards.append(np.mean(batch_reward))
        time_list.append(time.time() - it_time)

        print("Episode: {}, total numsteps: {}, reward: {}, time taken: {}".format(int(total_numsteps/env._max_episode_steps), total_numsteps, round(max_rewards[-1], 2), round(time_list[-1], 2)))

        # Rank the reward and normalize it
        batch_reward = torch.FloatTensor(normalized_rank(batch_reward))

        th_update = []
        optimizer.zero_grad()
        # for each actor's parameter, and for each noise in the batch, update it by the reward * the noise value
        for idx, p in enumerate(actor.parameters()):
            upd_weights = torch.zeros(p.data.shape)

            for n,r in zip(batch_noise, batch_reward):
                upd_weights += r * torch.Tensor(n[idx])

            upd_weights = upd_weights / (BATCH_SIZE*STD_NOISE)
            # put the updated weight on the gradient variable so that afterwards the optimizer will use it
            p.grad = torch.FloatTensor(-upd_weights).clone()

        # Optimize the actor's NN
        optimizer.step()

        if total_numsteps % args.log_interval == 0:
            save_results(args.env, args.seed, time_list, max_rewards, min_rewards, avg_rewards, total_start_time, updates, noise_mut)
            torch.save({'model_state_dict': agent.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                        checkpoint_name+'/actor_'+str(args.env)+'_'+str(args.seed)+'.pth.tar')

            if args.eval == True:
                eval(env, agent)

            # Anneal mutation in population
            std_loss = l1_loss(max(batch_reward).unsqueeze(0),torch.mean(batch_reward).unsqueeze(0)).unsqueeze(0)
            STD_NOISE = STD_NOISE + torch.clamp((args.lr_es / (BATCH_SIZE*STD_NOISE))*std_loss,0,args.clip)
            STD_NOISE = STD_NOISE[0]
            print(STD_NOISE)
            noise_mut.append(STD_NOISE)


                            ######################################### TERMINATE #############################################

    # quit the processes 
    for _ in range(args.workers):
        params_queue.put(None)

    for p in processes:
        p.join()

    #-------------------------------------------------------------------------------------------------------------------------------------------------------#





