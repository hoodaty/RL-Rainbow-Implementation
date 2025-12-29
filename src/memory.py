# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import torch

# HARDCODED FOR MINATAR (Breakout=4). Change to 6 for Space Invaders if needed.
CHANNELS = 4 

Transition_dtype = np.dtype([('timestep', np.int32), ('state', np.uint8, (CHANNELS, 10, 10)), ('action', np.int32), ('reward', np.float32), ('nonterminal', np.bool_)])
blank_trans = (0, np.zeros((CHANNELS, 10, 10), dtype=np.uint8), 0, 0.0, False)

class SegmentTree():
  def __init__(self, size):
    self.index = 0
    self.size = size
    self.full = False
    self.tree_start = 2**(size-1).bit_length()-1
    self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
    self.data = np.array([blank_trans] * size, dtype=Transition_dtype)
    self.max = 1

  def _update_nodes(self, indices):
    children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
    self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

  def _propagate(self, indices):
    parents = (indices - 1) // 2
    unique_parents = np.unique(parents)
    self._update_nodes(unique_parents)
    if parents[0] != 0:
      self._propagate(parents)

  def _propagate_index(self, index):
    parent = (index - 1) // 2
    left, right = 2 * parent + 1, 2 * parent + 2
    self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
    if parent != 0:
      self._propagate_index(parent)

  def update(self, indices, values):
    self.sum_tree[indices] = values
    self._propagate(indices)
    current_max_value = np.max(values)
    self.max = max(current_max_value, self.max)

  def _update_index(self, index, value):
    self.sum_tree[index] = value
    self._propagate_index(index)
    self.max = max(value, self.max)

  def append(self, data, value):
    self.data[self.index] = data
    self._update_index(self.index + self.tree_start, value)
    self.index = (self.index + 1) % self.size
    self.full = self.full or self.index == 0
    self.max = max(value, self.max)

  def _retrieve(self, indices, values):
    children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1))
    if children_indices[0, 0] >= self.sum_tree.shape[0]:
      return indices
    elif children_indices[0, 0] >= self.tree_start:
      children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
    left_children_values = self.sum_tree[children_indices[0]]
    successor_choices = np.greater(values, left_children_values).astype(np.int32)
    successor_indices = children_indices[successor_choices, np.arange(indices.size)]
    successor_values = values - successor_choices * left_children_values
    return self._retrieve(successor_indices, successor_values)

  def find(self, values):
    indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
    data_index = indices - self.tree_start
    return (self.sum_tree[indices], data_index, indices)

  def get(self, data_index):
    return self.data[data_index % self.size]

  def total(self):
    return self.sum_tree[0]

class ReplayMemory():
  def __init__(self, args, capacity):
    self.device = args.device
    self.capacity = capacity
    self.history = args.history_length
    self.discount = args.discount
    self.n = args.multi_step
    self.priority_weight = args.priority_weight
    self.priority_exponent = args.priority_exponent
    self.t = 0
    self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32, device=self.device)
    self.transitions = SegmentTree(capacity)

  def append(self, state, action, reward, terminal):
    # State is (History, C, 10, 10). We only store the newest frame: state[-1]
    # We store as uint8 to save space (though MinAtar is 0-1, scaling to 255 is safe)
    state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))
    self.transitions.append((self.t, state, action, reward, not terminal), self.transitions.max)
    self.t = 0 if terminal else self.t + 1

  def _get_transitions(self, idxs):
    transition_idxs = np.arange(-self.history + 1, self.n + 1) + np.expand_dims(idxs, axis=1)
    transitions = self.transitions.get(transition_idxs)
    transitions_firsts = transitions['timestep'] == 0
    blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
    for t in range(self.history - 2, -1, -1):
      blank_mask[:, t] = np.logical_or(blank_mask[:, t + 1], transitions_firsts[:, t + 1])
    for t in range(self.history, self.history + self.n):
      blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], transitions_firsts[:, t])
    transitions[blank_mask] = blank_trans
    return transitions

  def _get_samples_from_segments(self, batch_size, p_total):
    segment_length = p_total / batch_size
    segment_starts = np.arange(batch_size) * segment_length
    valid = False
    while not valid:
      samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts
      probs, idxs, tree_idxs = self.transitions.find(samples)
      if np.all((self.transitions.index - idxs) % self.capacity > self.n) and np.all((idxs - self.transitions.index) % self.capacity >= self.history) and np.all(probs != 0):
        valid = True
    
    transitions = self._get_transitions(idxs)
    all_states = transitions['state']
    
    # Get history frames. Shape: (Batch, History, C, 10, 10)
    states = torch.tensor(all_states[:, :self.history], device=self.device, dtype=torch.float32).div_(255)
    # Get next n frames
    next_states = torch.tensor(all_states[:, self.n:self.n + self.history], device=self.device, dtype=torch.float32).div_(255)

    actions = torch.tensor(np.copy(transitions['action'][:, self.history - 1]), dtype=torch.int64, device=self.device)
    rewards = torch.tensor(np.copy(transitions['reward'][:, self.history - 1:-1]), dtype=torch.float32, device=self.device)
    R = torch.matmul(rewards, self.n_step_scaling)
    nonterminals = torch.tensor(np.expand_dims(transitions['nonterminal'][:, self.history + self.n - 1], axis=1), dtype=torch.float32, device=self.device)
    
    return probs, idxs, tree_idxs, states, actions, R, next_states, nonterminals

  def sample(self, batch_size):
    p_total = self.transitions.total()
    probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = self._get_samples_from_segments(batch_size, p_total)
    probs = probs / p_total
    capacity = self.capacity if self.transitions.full else self.transitions.index
    weights = (capacity * probs) ** -self.priority_weight
    weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)
    return tree_idxs, states, actions, returns, next_states, nonterminals, weights

  def update_priorities(self, idxs, priorities):
    priorities = np.power(priorities, self.priority_exponent)
    self.transitions.update(idxs, priorities)

  def __iter__(self):
    self.current_idx = 0
    return self

  def __next__(self):
    if self.current_idx == self.capacity:
      raise StopIteration
    transitions = self.transitions.data[np.arange(self.current_idx - self.history + 1, self.current_idx + 1)]
    transitions_firsts = transitions['timestep'] == 0
    blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
    for t in reversed(range(self.history - 1)):
      blank_mask[t] = np.logical_or(blank_mask[t + 1], transitions_firsts[t + 1])
    transitions[blank_mask] = blank_trans
    state = torch.tensor(transitions['state'], dtype=torch.float32, device=self.device).div_(255)
    self.current_idx += 1
    return state

  next = __next__
