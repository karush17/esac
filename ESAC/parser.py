
import os, sys
import argparse

def build_parser():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env', default="HalfCheetah-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate of SAC (default: 0.0003)')
    parser.add_argument('--lr_es', type=float, default=0.005, metavar='G',
                        help='learning rate of ES (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=float, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--elite_rate', type=float, default=0.4, metavar='N',
                        help='Fraction of elites (default: 0.4)')
    parser.add_argument('--mutation', type=float, default=0.005, metavar='N',
                        help='Standard deviation of perturbations (default: 0.005)')
    parser.add_argument('--pop', type=int, default=10, metavar='N',
                        help='ES Population size (default: 10)')
    parser.add_argument('--grad_models', type=int, default=4, metavar='N',
                        help='Number of gradient model injections in population (default: 4)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 4)')
    parser.add_argument('--clip', type=float, default=1e-4, metavar='N',
                        help='clip parameter for AMT update (default: 1e-4)')
    parser.add_argument('--log_interval', type=int, default=20000, metavar='N',
                        help='save model and results every xth step (default: 20000)')

    return parser


