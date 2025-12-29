# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import pickle
import numpy as np
import torch
import math
from tqdm import trange

# Import from your custom distDQN files
from distDQNagent import Agent
from env import Env
from memory import ReplayMemory
from test import test

parser = argparse.ArgumentParser(description='Distributional DQN Baseline')
parser.add_argument('--id', type=str, default='dist_dqn_test', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='breakout', help='MinAtar game')
parser.add_argument('--T-max', type=int, default=int(1e6), metavar='STEPS', help='Number of training steps')
parser.add_argument('--max-episode-length', type=int, default=int(1000), metavar='LENGTH', help='Max episode length')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--hidden-size', type=int, default=64, metavar='SIZE', help='Network hidden size')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e5), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(1000), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=int(1000), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=5000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--checkpoint-interval', default=0, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file.')

# --- MISSING ARGUMENTS ADDED HERE ---
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')

# --- DISTRIBUTIONAL DQN SPECIFIC DEFAULTS ---
# 1. Multi-step = 1 (Standard C51)
parser.add_argument('--multi-step', type=int, default=1, metavar='n', help='Number of steps for multi-step return')
# 2. Priority Exponent = 0 (Uniform Replay)
parser.add_argument('--priority-exponent', type=float, default=0.0, metavar='ω', help='Prioritised experience replay exponent')
parser.add_argument('--priority-weight', type=float, default=0.0, metavar='β', help='Initial prioritised experience replay importance sampling weight')

args = parser.parse_args()

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))
results_dir = os.path.join('results', args.id)
if not os.path.exists(results_dir):
  os.makedirs(results_dir)
metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(np.random.randint(1, 10000))
else:
  args.device = torch.device('cpu')

def log(s):
  print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)

def load_memory(memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'rb') as pickle_file: return pickle.load(pickle_file)
  else:
    with bz2.open(memory_path, 'rb') as zipped_pickle_file: return pickle.load(zipped_pickle_file)

def save_memory(memory, memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'wb') as pickle_file: pickle.dump(memory, pickle_file)
  else:
    with bz2.open(memory_path, 'wb') as zipped_pickle_file: pickle.dump(memory, zipped_pickle_file)

env = Env(args)
env.train()
action_space = env.action_space()
args.n_channels = env.n_channels
dqn = Agent(args, env)

# CORRECTION: Ensure this uses args.model
if args.model is not None and not args.evaluate:
  if not args.memory: raise ValueError('Cannot resume training without memory save path.')
  mem = load_memory(args.memory, args.disable_bzip_memory)
else:
  mem = ReplayMemory(args, args.memory_capacity)

val_mem = ReplayMemory(args, args.evaluation_size)
T, done = 0, True
while T < args.evaluation_size:
  if done: state = env.reset()
  next_state, _, done = env.step(np.random.randint(0, action_space))
  val_mem.append(state, -1, 0.0, done)
  state = next_state
  T += 1

if args.evaluate:
  dqn.eval()
  avg_reward, avg_Q = test(args, 0, dqn, val_mem, metrics, results_dir, evaluate=True)
  print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
else:
  dqn.train()
  done = True
  
  # --- Epsilon Decay Setup ---
  eps_start = 1.0
  eps_end = 0.01
  eps_decay = 50000 # Decay over 50k steps (MinAtar scale)

  for T in trange(1, args.T_max + 1):
    if done:
      state = env.reset()

    # Calculate Epsilon
    epsilon = eps_end + (eps_start - eps_end) * math.exp(-1. * T / eps_decay)

    # Use Epsilon-Greedy (Available in agent.py)
    action = dqn.act_e_greedy(state, epsilon)
    
    next_state, reward, done = env.step(action)
    if args.reward_clip > 0:
      reward = max(min(reward, args.reward_clip), -args.reward_clip)
    mem.append(state, action, reward, done)

    if T >= args.learn_start:
      if T % args.replay_frequency == 0:
        dqn.learn(mem)

      if T % args.evaluation_interval == 0:
        dqn.eval()
        avg_reward, avg_Q = test(args, T, dqn, val_mem, metrics, results_dir)
        log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q) + ' | Epsilon: ' + str(round(epsilon, 3)))
        dqn.train()
        if args.memory is not None: save_memory(mem, args.memory, args.disable_bzip_memory)

      if T % args.target_update == 0:
        dqn.update_target_net()

      if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
        dqn.save(results_dir, 'checkpoint.pth')

    state = next_state

env.close()
