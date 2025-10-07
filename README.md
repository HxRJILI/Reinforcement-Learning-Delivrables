# Grid World Reinforcement Learning 

A comprehensive implementation of Q-Learning and Deep Q-Network (DQN) agents for configurable grid world environments, featuring training, evaluation, convergence analysis, and hyperparameter sensitivity studies.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Algorithms](#algorithms)
- [Usage Guide](#usage-guide)
- [Results](#results)
- [Configuration](#configuration)

## 🎯 Overview

This project implements flexible Grid World environments with two reinforcement learning algorithms:

**Environment Features:**
- Configurable grid sizes (5×5 to 15×15+)
- Multiple goals and obstacles
- Stochastic dynamics (action noise, slipping)
- Customizable reward structures
- Real-time visualization

**Implemented Algorithms:**
- **Tabular Q-Learning** - Classical RL approach
- **Deep Q-Network (DQN)** - Neural network-based value learning with experience replay

## ✨ Features

### Environment (`gridworld_env.py`)
- ✅ Gymnasium-style API
- ✅ Multiple goals and obstacles
- ✅ Stochastic transitions
- ✅ Real-time visualization
- ✅ Episode truncation limits

### Q-Learning Agent (`qlearning_agent.py`)
- ✅ Tabular Q-learning
- ✅ Epsilon-greedy exploration
- ✅ Automatic parameter decay
- ✅ Save/load functionality

### DQN Agent (`dqn_agent.py`)
- ✅ Deep Q-Network with target network
- ✅ Experience replay buffer
- ✅ Configurable network architecture
- ✅ Multi-environment training pipeline
- ✅ Comprehensive visualization and analysis
- ✅ Path tracking and PNG exports

### Analysis Tools
- ✅ Convergence analysis across grid sizes
- ✅ Hyperparameter sensitivity analysis
- ✅ Moving goal experiments (demonstrates Q-learning limitations)
- ✅ Training metrics and plots

## 📦 Installation

### Prerequisites
```bash
Python 3.8+
```

### Dependencies
```bash
pip install numpy matplotlib pandas seaborn torch
```

### Clone Repository
```bash
git clone <repository-url>
cd grid-world-rl
```

## 🚀 Quick Start

### Train Q-Learning Agents
```bash
# Train on multiple grid sizes
python grid_convergence_analysis.py --mode train

# Test a trained agent
python grid_convergence_analysis.py --mode inference --grid-size 10
```

### Train DQN Agents
```bash
# Train DQN on three different environments
python dqn_agent.py
```

This will:
- Train agents on Simple 5×5, Obstacles 8×8, and Multi-Goal 10×10 environments
- Save trained models to `DQN_Agent_Analysis/`
- Generate training curves and performance plots
- Visualize agent behavior with path tracking
- Export paths as PNG images

### Hyperparameter Sensitivity Analysis
```bash
python HyperparameterSensitivity.py
```

### Moving Goal Experiment
```bash
python moving_goal_experiment.py
```

Demonstrates why tabular Q-learning fails in non-stationary environments.

## 📁 Project Structure

```
grid-world-rl/
│
├── gridworld_env.py                   # Grid World environment
├── qlearning_agent.py                 # Q-Learning implementation
├── dqn_agent.py                       # DQN implementation
├── grid_convergence_analysis.py       # Q-Learning convergence analysis
├── HyperparameterSensitivity.py      # Hyperparameter tuning
├── moving_goal_experiment.py          # Non-stationary experiments
├── README.md
│
├── trained_agents/                    # Q-Learning agents
│   ├── agent_grid5x5.pkl
│   └── ...
│
├── DQN_Agent_Analysis/                # DQN results
│   ├── agent_Simple_5x5.pth          # Trained DQN models
│   ├── training_stats_*.pkl          # Training metrics
│   ├── path_*_ep*.png                # Path visualizations
│   ├── path_*_ep*.pkl                # Path data
│   ├── training_results.png          # Training curves
│   └── results_summary.json          # Performance summary
│
├── results/                           # Q-Learning analysis
│   ├── convergence_rewards.png
│   ├── convergence_steps.png
│   └── ...
│
├── sensitivity_analysis/              # Hyperparameter results
│   ├── sensitivity_learning_rate.png
│   └── ...
│
└── Moving_Goal_Results/               # Moving goal experiments
    ├── comprehensive_comparison.png
    ├── q_value_heatmaps.png
    └── experiment_report.txt
```

## 🤖 Algorithms

### Tabular Q-Learning

Classic reinforcement learning using a state-action value table.

**Pros:**
- Simple and interpretable
- Guaranteed convergence (stationary environments)
- No hyperparameter tuning for network architecture

**Cons:**
- Doesn't scale to large state spaces
- No generalization between states
- Fails in non-stationary environments

**Best for:** Small grids (5×5 to 10×10), educational purposes

### Deep Q-Network (DQN)

Neural network approximation of Q-values with experience replay.

**Features:**
- **Experience Replay**: Breaks correlation between consecutive samples
- **Target Network**: Stabilizes training with periodic updates
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation
- **Configurable Architecture**: Adjust network size per environment complexity

**Architecture:**
```
Input (2D state) → Hidden Layers → Output (4 Q-values for actions)
```

**Hyperparameters:**
- Simple 5×5: [64, 64] hidden units, 500 episodes
- Obstacles 8×8: [128, 128] hidden units, 1000 episodes
- Multi-Goal 10×10: [256, 256] hidden units, 1500 episodes

**Pros:**
- Scales to larger state spaces
- Generalizes between similar states
- More sample efficient in complex environments

**Cons:**
- Requires more tuning
- Longer training time
- Less interpretable

**Best for:** Complex grids (8×8+), multi-goal tasks, future extension to pixels

## 📖 Usage Guide

### Q-Learning Training

```bash
# Train on multiple grid sizes
python grid_convergence_analysis.py --mode train --num-episodes 2000

# Test specific agent
python grid_convergence_analysis.py --mode inference --grid-size 10

# Compare all agents
python grid_convergence_analysis.py --mode compare
```

### DQN Training

The DQN script automatically trains on three environments:

**1. Simple 5×5** - Basic navigation
- Grid: 5×5
- Goal: Single goal at (4,4)
- Obstacles: None
- Episodes: 500

**2. Obstacles 8×8** - Navigation with obstacles
- Grid: 8×8
- Goal: Single goal at (7,7)
- Obstacles: 6 obstacles forming barriers
- Episodes: 1000

**3. Multi-Goal 10×10** - Complex multi-goal task
- Grid: 10×10
- Goals: 4 goals at corners
- Obstacles: 5 obstacles around center
- Episodes: 1500

**Output Files:**
- `agent_[env_name].pth` - Trained models
- `training_stats_[env_name].pkl` - Metrics (rewards, lengths, success rate, loss)
- `path_[env_name]_ep[N].png` - Path visualizations (3 per environment)
- `path_[env_name]_ep[N].pkl` - Path data (positions, actions, rewards)
- `training_results.png` - 4-panel training comparison
- `results_summary.json` - Final performance metrics

### Hyperparameter Sensitivity

```bash
python HyperparameterSensitivity.py
```

Tests: learning rate, discount factor, epsilon decay

**Output:**
- Individual parameter plots
- Comparison plot
- Text report with recommendations

### Moving Goal Experiment

```bash
python moving_goal_experiment.py
```

**Purpose:** Demonstrates that tabular Q-learning fails when goals change between episodes.

**Results:**
- Success rate drops by ~67%
- Q-value heatmaps show confused policy
- Comprehensive comparison plots

## 📊 Results

### Q-Learning Performance (1000 episodes)

| Grid Size | Success Rate | Avg Steps | Q-Table Size |
|-----------|--------------|-----------|--------------|
| 5×5       | 94.2%        | 11.3      | 25           |
| 7×7       | 91.8%        | 17.9      | 49           |
| 10×10     | 88.4%        | 28.6      | 100          |
| 12×12     | 85.1%        | 36.2      | 144          |
| 15×15     | 81.7%        | 49.8      | 225          |

### DQN Performance

| Environment | Success Rate | Avg Reward | Avg Steps |
|-------------|--------------|------------|-----------|
| Simple 5×5  | ~95%         | 9.5+       | 12-15     |
| Obstacles 8×8 | ~90%       | 18+        | 25-30     |
| Multi-Goal 10×10 | ~85%    | 60+        | 40-50     |

*Note: DQN performance improves with training. Final values from last 100 episodes.*

### Moving Goal Degradation

| Metric | Fixed Goal | Moving Goal | Degradation |
|--------|-----------|-------------|-------------|
| Success Rate | 91.2% | 23.7% | **-67.5%** |
| Avg Reward | 8.45 | -4.21 | **-12.66** |
| Avg Steps | 15.3 | 89.6 | **+74.3** |

**Conclusion:** Standard Q-learning fails in non-stationary environments. Goal-conditioned methods (HER, UVFA) or meta-learning are required.

## ⚙️ Configuration

### Environment Configuration

```python
from gridworld_env import ConfigurableGridWorld, GridWorldConfig

config = GridWorldConfig(
    grid_size=8,
    start_pos=(0, 0),
    goals=[(7, 7)],
    obstacles=[(3, 3), (4, 4)],
    goal_reward=10.0,
    step_penalty=-0.01,
    obstacle_penalty=-1.0,
    max_steps=100
)

env = ConfigurableGridWorld(config=config, render_mode="human")
```

### Q-Learning Agent

```python
from qlearning_agent import QLearningAgent

agent = QLearningAgent(
    num_actions=4,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.995
)
```

### DQN Agent

```python
from dqn_agent import DQNAgent

agent = DQNAgent(
    state_size=2,
    action_size=4,
    learning_rate=0.001,
    gamma=0.99,
    epsilon_decay=0.995,
    buffer_size=20000,
    batch_size=64,
    hidden_sizes=[128, 128]
)
```

### Customizing DQN Training

Edit `dqn_agent.py` main() function:

```python
# Add your custom environment
environments = {
    'My_Custom_Env': {
        'config': GridWorldConfig(
            grid_size=12,
            start_pos=(0, 0),
            goals=[(11, 11)],
            obstacles=[(5, 5), (6, 6)],
            goal_reward=15.0,
            step_penalty=-0.02,
            max_steps=150
        ),
        'episodes': 1200,
        'hidden_sizes': [256, 128],
        'learning_rate': 0.0005,
        'epsilon_decay': 0.997
    }
}
```

## 💡 Examples

### Example 1: Train Q-Learning Agent

```python
from gridworld_env import ConfigurableGridWorld, GridWorldConfig
from qlearning_agent import QLearningAgent, train_agent

config = GridWorldConfig(
    grid_size=5,
    start_pos=(0, 0),
    goals=[(4, 4)],
    goal_reward=1.0
)

env = ConfigurableGridWorld(config=config)
agent = QLearningAgent(num_actions=4)

metrics = train_agent(env, agent, num_episodes=1000)
agent.save("my_agent.pkl")
```

### Example 2: Train DQN Agent

```python
from dqn_agent import DQNAgent, train_agent
from gridworld_env import ConfigurableGridWorld, GridWorldConfig

config = GridWorldConfig(grid_size=8, goals=[(7, 7)])
env = ConfigurableGridWorld(config=config)

agent = DQNAgent(state_size=2, action_size=4, hidden_sizes=[128, 128])
train_agent(env, agent, num_episodes=1000, save_dir="my_results", env_name="test")
```

### Example 3: Load and Test DQN Agent

```python
from dqn_agent import DQNAgent
from gridworld_env import ConfigurableGridWorld, GridWorldConfig
import time

# Load agent
agent = DQNAgent(state_size=2, action_size=4)
agent.load("DQN_Agent_Analysis/agent_Simple_5x5.pth")

# Create environment
config = GridWorldConfig(grid_size=5, goals=[(4, 4)])
env = ConfigurableGridWorld(config=config, render_mode="human")

# Test agent
state, _ = env.reset()
for _ in range(50):
    action = agent.select_action(state, training=False)
    state, reward, done, truncated, _ = env.step(action)
    time.sleep(0.2)
    if done or truncated:
        break

env.close()
```

## 🔬 Key Findings

### Algorithm Comparison

**Tabular Q-Learning:**
- Faster training on small grids
- Perfect state representation
- Fails completely with non-stationary goals
- Memory grows O(n²) with grid size

**DQN:**
- Better generalization
- Scales to larger grids
- More robust to environment variations
- Requires more training episodes
- Could potentially handle goal-conditioning with architectural changes

### Hyperparameter Sensitivity

- **Learning Rate**: Most critical (0.1-0.3 optimal for Q-learning, 0.0003-0.001 for DQN)
- **Discount Factor**: High values (0.95-0.99) necessary
- **Epsilon Decay**: Slower decay improves exploration
- **Network Size (DQN)**: Larger networks for complex environments

### Non-Stationary Environments

**Critical Finding:** Both tabular Q-learning and standard DQN fail when goals change randomly between episodes.

**Why:**
- Stationarity assumption violated
- Conflicting Q-value updates
- No goal representation in state
- Catastrophic forgetting

**Solutions Required:**
- Goal-conditioned RL (Q(s, g, a))
- Hindsight Experience Replay (HER)
- Universal Value Function Approximators (UVFA)
- Meta-learning approaches

## 🛠️ Troubleshooting

### "Agent file not found"
Run training first: `python grid_convergence_analysis.py --mode train`

### DQN not learning on complex environments
- Increase training episodes (try 1500-2000)
- Use larger network ([256, 256] hidden units)
- Lower learning rate (0.0003)
- Slower epsilon decay (0.997-0.998)

### Out of memory errors
- Reduce replay buffer size
- Use smaller network
- Train on smaller grids first

### Agent learns slowly
- Run hyperparameter sensitivity analysis
- Check reward structure (goal reward should be >> step penalty)
- Ensure obstacles aren't blocking all paths

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 📄 License

MIT License - Open source and free to use.

## 🙏 Acknowledgments

- NumPy, Matplotlib, PyTorch, Pandas, Seaborn
- Inspired by OpenAI Gym/Gymnasium
- Q-Learning: Watkins & Dayan (1992)
- DQN: Mnih et al. (2015)

---
**BY: RJILI HOUSSAM**
**Version:** 2.1  
**Last Updated:** 10/7/2025