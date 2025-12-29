# Rainbow & Distributional DQN on MinAtar

This repository contains a PyTorch implementation of the **Rainbow** agent (Hessel et al., 2018) and a **Distributional DQN** baseline, adapted for the **MinAtar** environment.

The project replicates the key components of the Rainbow paper to verify claims of data efficiency and performance on a computationally constrained environment (MinAtar), which serves as a miniature version of the Atari 2600 benchmark.

Authors:
* Alice Lataste
* Sami Laita
* Soumodeep Hoodaty

## 🔬 Architecture & Adaptations

This implementation adapts the original Atari architecture for MinAtar's $10 \times 10 \times N$ channel inputs:

* **Rainbow Agent:** Integrates **Dueling Networks**, **Noisy Nets** (for exploration), **Distributional RL (C51)**, **Multi-step Returns** ($n=3$), and **Prioritized Experience Replay**.
* **Distributional DQN (Baseline):** A stripped-down version using only Distributional RL (C51), $\epsilon$-greedy exploration, standard Linear layers, 1-step returns, and Uniform Replay.
* **Input Adaptation:** The standard DeepMind CNN is replaced with a lightweight CNN (Kernel 3x3) to handle the $10 \times 10$ grid input of MinAtar games like *Breakout*.

## ⚙️ Installation

To run this code, you need Python 3, PyTorch, and the MinAtar environment.

```bash
# Install PyTorch (select the version appropriate for your CUDA setup)
pip install torch torchvision

# Install MinAtar
pip install minatar

# Install other dependencies
pip install numpy tqdm matplotlib

```

## 🚀 Usage

### 1. Train the Rainbow Agent

Run the full Rainbow agent using `main.py`. This uses Noisy Nets for exploration (no epsilon).

```bash
python main.py --game breakout --id rainbow_run --multi-step 3 --atoms 51

```

### 2. Train the Distributional DQN Baseline

Run the baseline using `distDQNmain.py`. This uses -greedy exploration and disables Rainbow features (Dueling, Priority, Multi-step).

```bash
python distDQNmain.py --game breakout --id dist_dqn_baseline

```

### 3. Compare Results

Use the provided plotting script to visualize the performance difference between the two agents.

```python
# Save this in a .py file or run in a notebook
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load metrics
r_steps, r_rewards = torch.load('results/rainbow_run/metrics.pth')['steps'], torch.load('results/rainbow_run/metrics.pth')['rewards']
d_steps, d_rewards = torch.load('results/dist_dqn_baseline/metrics.pth')['steps'], torch.load('results/dist_dqn_baseline/metrics.pth')['rewards']

# Plot
plt.plot(r_steps, [np.mean(r) for r in r_rewards], label='Rainbow')
plt.plot(d_steps, [np.mean(r) for r in d_rewards], label='Distributional DQN', linestyle='--')
plt.legend()
plt.show()

```

## 📂 File Structure

* **`main.py`**: Training loop for the Rainbow agent.
* **`distDQNmain.py`**: Training loop for the Distributional DQN baseline (handles -decay).
* **`model.py`**: Contains the `DQN` class. **Note:** This file must be manually toggled/overwritten depending on which agent you are running (Rainbow requires NoisyLayers; DistDQN requires standard Linear layers).
* **`agent.py`**: The Rainbow learning logic (Distributional Loss + Dueling aggregation).
* **`distDQNagent.py`**: The Baseline learning logic (Distributional Loss only).
* **`env.py`**: Wrapper to convert MinAtar boolean grids to PyTorch float tensors.
* **`memory.py`**: Prioritized Experience Replay buffer (with N-step support).
* **`test.py`**: Evaluation loop.

## 📚 References

This project is a replication study based on the following paper and code:

**1. The Paper**

> Hessel, M., Modayil, J., van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., Horgan, D., Piot, B., Azar, M., & Silver, D. (2018). **Rainbow: Combining Improvements in Deep Reinforcement Learning**. *Thirty-Second AAAI Conference on Artificial Intelligence*.

**2. Original Implementation**

> **Kaixhin/Rainbow**: https://github.com/Kaixhin/Rainbow
> *This repository served as the foundation for the code structure, which was then modified for MinAtar compatibility.*

```

```
