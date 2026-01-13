# Rainbow & Distributional DQN on MinAtar

This repository contains a PyTorch implementation of the **Rainbow** agent (Hessel et al., 2018) and a **Distributional DQN** baseline, adapted for the **MinAtar** environment.

The project replicates the key components of the Rainbow paper to verify claims of data efficiency and performance on a computationally constrained environment (MinAtar). It includes rigorous ablation studies and experimentation with replay frequency to test the agent's robustness.

**Authors:**
* Alice Lataste
* Sami Laita
* Soumodeep Hoodaty

## 🔬 Architecture & Adaptations

This implementation adapts the original Atari architecture for MinAtar's $10 \times 10 \times N$ channel inputs:

* **Rainbow Agent:** Integrates **Dueling Networks**, **Noisy Nets** (for exploration), **Distributional RL (C51)**, **Multi-step Returns** ($n=3$), and **Prioritized Experience Replay**.
* **Distributional DQN (Baseline):** A stripped-down version using only Distributional RL (C51), $\epsilon$-greedy exploration, standard Linear layers, 1-step returns, and Uniform Replay.
* **Input Adaptation:** The standard DeepMind CNN is replaced with a lightweight CNN (Kernel 3x3) to handle the $10 \times 10$ grid input.

## ⚙️ Installation

To run this code, you need Python 3, PyTorch, MinAtar, and Weights & Biases (for logging).

```bash
# Install PyTorch
pip install torch torchvision

# Install MinAtar and dependencies
pip install minatar numpy tqdm matplotlib imageio

# Install Weights & Biases
pip install wandb
wandb login

```

## 🚀 Usage & Experiments

### 1. Train the Rainbow Agent (Standard)

Run the full Rainbow agent with WandB logging enabled.

```bash
python main.py --game breakout --id rainbow_breakout --wandb --wandb-project "rainbow-minatar"

```

### 2. Train the Distributional DQN Baseline

Run the baseline (C51 only) to compare against Rainbow.

```bash
python distDQNmain.py --game breakout --id dist_dqn_baseline --wandb --wandb-project "rainbow-minatar"

```

### 3. Ablation Studies

We verify the paper's claims by removing specific components from Rainbow to observe performance drops.

* **No Prioritized Replay (Uniform Sampling):**
```bash
python main.py --game breakout --id rainbow_no_prio --priority-exponent 0.0 --priority-weight 0.0 --wandb

```


* **No Multi-step Returns (1-step only):**
*Crucial for complex games like Seaquest.*
```bash
python main.py --game seaquest --id rainbow_no_multistep --multi-step 1 --wandb

```



### 4. Replay Frequency Experiment

We investigated the effect of training frequency on data efficiency. Increasing the update frequency (learning every 2 frames instead of 4) significantly improved the Area Under the Learning Curve (ALC).

```bash
# Train every 2 frames (High Frequency)
python main.py --game breakout --id rainbow_freq2 --replay-frequency 2 --wandb

```

## 🎥 Visualizations

You can generate a gameplay video (GIF) of your trained agent using the `record_video.py` script.

```bash
# Generate video for a trained Breakout agent
python record_video.py --game breakout --model results/rainbow_breakout/model.pth

# Generate video for Seaquest
python record_video.py --game seaquest --model results/rainbow_seaquest/model.pth

```

*The output GIF will be saved in the current directory (e.g., `breakout_gameplay.gif`).*

## 📂 File Structure

* **`main.py`**: Training loop for the Rainbow agent (WandB integrated).
* **`distDQNmain.py`**: Training loop for the Distributional DQN baseline.
* **`model.py`**: Rainbow Architecture (NoisyLayers + Dueling).
* **`distDQNmodel.py`**: Baseline Architecture (Linear + C51 Head only).
* **`agent.py`**: Rainbow learning logic (N-step + Priority updates).
* **`distDQNagent.py`**: Baseline learning logic (1-step + Uniform updates).
* **`memory.py`**: Prioritized Experience Replay buffer (handles N-step calculations).
* **`env.py`**: Wrapper for MinAtar state conversion and frame stacking.
* **`record_video.py`**: Script to record and save gameplay GIFs.

## 📚 References

**1. The Paper**

> Hessel, M., et al. (2018). **Rainbow: Combining Improvements in Deep Reinforcement Learning**. *AAAI Conference on Artificial Intelligence*.

**2. Original Implementation**

> **Kaixhin/Rainbow**: https://github.com/Kaixhin/Rainbow

```

```
