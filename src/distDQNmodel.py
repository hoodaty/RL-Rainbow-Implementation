# -*- coding: utf-8 -*-
from __future__ import division
import torch
from torch import nn
from torch.nn import functional as F

class DQN(nn.Module):
  def __init__(self, args, action_space):
    super(DQN, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space
    
    # MinAtar Adaptation: 
    # Check if 'n_channels' is in args (passed from Env), otherwise assume 4 (Breakout)
    n_channels = getattr(args, 'n_channels', 4) 
    self.input_channels = args.history_length * n_channels

    # Standard CNN for MinAtar (same as Rainbow adaptation)
    self.convs = nn.Sequential(
        nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1), 
        nn.ReLU()
    )
    self.conv_output_size = 16 * 8 * 8

    # Distributional DQN (C51) Architecture:
    # 1. Standard Linear Layer (Noisy removed)
    # 2. Output is (Action_Space x Atoms) directly (No Dueling Split)
    self.fc_h = nn.Linear(self.conv_output_size, args.hidden_size)
    self.fc_z = nn.Linear(args.hidden_size, action_space * self.atoms)

  def forward(self, x, log=False):
    x = x.view(-1, self.input_channels, 10, 10)
    x = self.convs(x)
    x = x.view(-1, self.conv_output_size)
    
    # Standard Head
    x = F.relu(self.fc_h(x))
    x = self.fc_z(x)
    
    # Reshape to (Batch, Actions, Atoms)
    q = x.view(-1, self.action_space, self.atoms)
    
    if log:
      q = F.log_softmax(q, dim=2)
    else:
      q = F.softmax(q, dim=2)
    return q

  def reset_noise(self):
    # No noise in standard Distributional DQN
    pass
