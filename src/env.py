# -*- coding: utf-8 -*-
from collections import deque
import torch
import numpy as np
from minatar import Environment

class Env():
  def __init__(self, args):
    self.device = args.device
    self.env = Environment(args.game)
    self.num_actions = self.env.num_actions()
    
    # MinAtar state is (10, 10, C). We need to verify channels.
    self.n_channels = self.env.state_shape()[2]
    
    # History buffer for stacking frames
    self.window = args.history_length
    self.state_buffer = deque([], maxlen=self.window)
    self.training = True 

  def _get_state(self):
    # Convert MinAtar (10, 10, C) boolean -> PyTorch (C, 10, 10) float
    s = self.env.state()
    return torch.tensor(s, dtype=torch.float32, device=self.device).permute(2, 0, 1)

  def _reset_buffer(self):
    # Fill buffer with blank states
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(self.n_channels, 10, 10, device=self.device))

  def reset(self):
    self.env.reset()
    self._reset_buffer()
    # Get initial state
    observation = self._get_state()
    self.state_buffer.append(observation)
    # Return stacked history: (History, C, 10, 10)
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    reward, done = self.env.act(action)
    observation = self._get_state()
    self.state_buffer.append(observation)
    # Return stacked history, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done

  def train(self):
    self.training = True

  def eval(self):
    self.training = False

  def action_space(self):
    return self.num_actions

  def render(self):
    pass # MinAtar visualizer doesn't work easily in Colab without specific setups

  def close(self):
    pass
