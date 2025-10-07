import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import pickle
import json
import os
from pathlib import Path
from typing import List, Tuple
import time

# Import the environment (assuming it's in gridworld_env.py)
from configurable_gridworld import ConfigurableGridWorld, GridWorldConfig


# Transition tuple for replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class DQNNetwork(nn.Module):
    """Deep Q-Network for Grid World."""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [128, 128]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience Replay Buffer."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        """Save a transition."""
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        states = torch.FloatTensor([t.state for t in batch])
        actions = torch.LongTensor([t.action for t in batch])
        next_states = torch.FloatTensor([t.next_state for t in batch])
        rewards = torch.FloatTensor([t.reward for t in batch])
        dones = torch.FloatTensor([t.done for t in batch])
        
        return states, actions, next_states, rewards, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for Grid World."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        hidden_sizes: List[int] = [128, 128]
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Networks
        self.policy_net = DQNNetwork(state_size, action_size, hidden_sizes)
        self.target_net = DQNNetwork(state_size, action_size, hidden_sizes)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training stats
        self.episode_count = 0
        self.training_step = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, next_state, reward, done):
        """Store transition in replay buffer."""
        self.memory.push(state, action, next_state, reward, done)
    
    def train_step(self):
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_step += 1
        
        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save agent."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'training_step': self.training_step
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent."""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        self.training_step = checkpoint['training_step']


def train_agent(
    env: ConfigurableGridWorld,
    agent: DQNAgent,
    num_episodes: int,
    save_dir: str,
    env_name: str
):
    """Train the DQN agent."""
    
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    losses = []
    
    print(f"\n{'='*60}")
    print(f"Training on: {env_name}")
    print(f"{'='*60}")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated):
            action = agent.select_action(state, training=True)
            next_state, reward, done, truncated, _ = env.step(action)
            
            agent.store_transition(state, action, next_state, reward, done or truncated)
            
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        agent.decay_epsilon()
        agent.episode_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        success_rate.append(1.0 if done else 0.0)
        
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            avg_success = np.mean(success_rate[-50:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Success Rate: {avg_success:.2%} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    # Save agent
    agent_path = os.path.join(save_dir, f'agent_{env_name}.pth')
    agent.save(agent_path)
    print(f"\n✓ Agent saved to {agent_path}")
    
    # Save training stats
    stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rate': success_rate,
        'losses': losses,
        'env_name': env_name
    }
    
    stats_path = os.path.join(save_dir, f'training_stats_{env_name}.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    
    return stats


def evaluate_agent(
    env: ConfigurableGridWorld,
    agent: DQNAgent,
    num_episodes: int = 10
):
    """Evaluate agent performance."""
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated):
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, _ = env.step(action)
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        if done:
            success_count += 1
    
    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / num_episodes
    }


def visualize_agent(
    env: ConfigurableGridWorld,
    agent: DQNAgent,
    save_dir: str,
    env_name: str,
    num_episodes: int = 3
):
    """Visualize agent's behavior and save paths."""
    
    print(f"\n{'='*60}")
    print(f"Visualizing agent on: {env_name}")
    print(f"{'='*60}")
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=episode)
        path = [state.copy()]
        actions_taken = []
        done = False
        truncated = False
        total_reward = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        while not (done or truncated):
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, info = env.step(action)
            
            path.append(next_state.copy())
            actions_taken.append(action)
            total_reward += reward
            state = next_state
            
            time.sleep(0.3)  # Slow down for visualization
        
        print(f"  Steps: {len(path) - 1}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Success: {'Yes' if done else 'No'}")
        
        # Save path data
        path_data = {
            'episode': episode,
            'path': path,
            'actions': actions_taken,
            'total_reward': total_reward,
            'success': done,
            'steps': len(path) - 1
        }
        
        path_file = os.path.join(save_dir, f'path_{env_name}_ep{episode}.pkl')
        with open(path_file, 'wb') as f:
            pickle.dump(path_data, f)
        
        # Create PNG visualization of the path
        plot_path(env.config, path, save_dir, env_name, episode, total_reward, done)
    
    env.close()


def plot_path(
    config: GridWorldConfig,
    path: List[np.ndarray],
    save_dir: str,
    env_name: str,
    episode: int,
    total_reward: float,
    success: bool
):
    """Create and save a PNG visualization of the agent's path."""
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw grid
    for i in range(config.grid_size + 1):
        ax.plot([0, config.grid_size], [i, i], 'k-', linewidth=0.5, alpha=0.3)
        ax.plot([i, i], [0, config.grid_size], 'k-', linewidth=0.5, alpha=0.3)
    
    # Draw obstacles
    for obs in config.obstacles:
        obs_rect = plt.Rectangle(
            (obs[1], config.grid_size - 1 - obs[0]),
            1, 1,
            linewidth=1,
            edgecolor='black',
            facecolor='black',
            alpha=0.8
        )
        ax.add_patch(obs_rect)
    
    # Draw goals
    for goal in config.goals:
        goal_rect = plt.Rectangle(
            (goal[1], config.grid_size - 1 - goal[0]),
            1, 1,
            linewidth=2,
            edgecolor='red',
            facecolor='lightcoral',
            alpha=0.7
        )
        ax.add_patch(goal_rect)
    
    # Draw path
    path_array = np.array(path)
    path_x = path_array[:, 1] + 0.5
    path_y = config.grid_size - 1 - path_array[:, 0] + 0.5
    
    # Plot path line
    ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.6, label='Path', zorder=5)
    
    # Plot path points
    colors = plt.cm.viridis(np.linspace(0, 1, len(path)))
    for i, (x, y) in enumerate(zip(path_x, path_y)):
        ax.scatter(x, y, c=[colors[i]], s=100, alpha=0.8, edgecolors='black', 
                  linewidth=1, zorder=10)
    
    # Mark start and end
    start_circle = plt.Circle(
        (path_x[0], path_y[0]),
        0.4,
        color='green',
        alpha=0.9,
        zorder=15,
        label='Start'
    )
    ax.add_patch(start_circle)
    
    end_marker = 'o' if success else 'X'
    end_color = 'darkgreen' if success else 'darkred'
    ax.scatter(path_x[-1], path_y[-1], c=end_color, s=400, marker=end_marker, 
              alpha=0.9, edgecolors='black', linewidth=2, zorder=20, label='End')
    
    # Add step numbers at key points
    step_interval = max(1, len(path) // 10)
    for i in range(0, len(path), step_interval):
        ax.text(path_x[i], path_y[i], str(i), fontsize=8, ha='center', 
               va='center', fontweight='bold', color='white', zorder=25)
    
    # Configuration
    ax.set_xlim(0, config.grid_size)
    ax.set_ylim(0, config.grid_size)
    ax.set_aspect('equal')
    
    status = "SUCCESS ✓" if success else "FAILED ✗"
    title = f'{env_name} - Episode {episode + 1} - {status}\n'
    title += f'Steps: {len(path) - 1} | Total Reward: {total_reward:.2f}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_xticks(range(config.grid_size + 1))
    ax.set_yticks(range(config.grid_size + 1))
    ax.grid(False)
    
    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    
    plt.tight_layout()
    
    # Save
    path_png = os.path.join(save_dir, f'path_{env_name}_ep{episode}.png')
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Path visualization saved: {path_png}")


def plot_training_results(save_dir: str):
    """Plot training results from all environments."""
    
    stats_files = [f for f in os.listdir(save_dir) if f.startswith('training_stats_')]
    
    if not stats_files:
        print("No training stats found!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Training Results', fontsize=16, fontweight='bold')
    
    for stats_file in stats_files:
        with open(os.path.join(save_dir, stats_file), 'rb') as f:
            stats = pickle.load(f)
        
        env_name = stats['env_name']
        
        # Rewards
        window = 50
        smoothed_rewards = np.convolve(stats['episode_rewards'], 
                                      np.ones(window)/window, mode='valid')
        axes[0, 0].plot(smoothed_rewards, label=env_name, linewidth=2)
        
        # Episode lengths
        smoothed_lengths = np.convolve(stats['episode_lengths'], 
                                      np.ones(window)/window, mode='valid')
        axes[0, 1].plot(smoothed_lengths, label=env_name, linewidth=2)
        
        # Success rate
        smoothed_success = np.convolve(stats['success_rate'], 
                                      np.ones(window)/window, mode='valid')
        axes[1, 0].plot(smoothed_success, label=env_name, linewidth=2)
        
        # Loss
        if stats['losses']:
            axes[1, 1].plot(stats['losses'], label=env_name, linewidth=2, alpha=0.7)
    
    axes[0, 0].set_title('Episode Rewards (Smoothed)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Episode Lengths (Smoothed)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Success Rate (Smoothed)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Training Loss')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Training results plot saved to {save_dir}/training_results.png")
    plt.show()


def main():
    """Main training and evaluation pipeline."""
    
    # Create analysis directory
    save_dir = 'DQN_Agent_Analysis'
    Path(save_dir).mkdir(exist_ok=True)
    print(f"Analysis directory: {save_dir}")
    
    # Define different environments
    environments = {
        'Simple_5x5': GridWorldConfig(
            grid_size=5,
            start_pos=(0, 0),
            goals=[(4, 4)],
            obstacles=[],
            goal_reward=10.0,
            step_penalty=-0.01,
            max_steps=50
        ),
        
        'Obstacles_8x8': GridWorldConfig(
            grid_size=8,
            start_pos=(0, 0),
            goals=[(7, 7)],
            obstacles=[(3, 3), (3, 4), (4, 3), (4, 4), (2, 5), (5, 2)],
            goal_reward=20.0,
            step_penalty=-0.05,
            obstacle_penalty=-2.0,
            max_steps=100
        ),
        
        'Multi_Goal_10x10': GridWorldConfig(
            grid_size=10,
            start_pos=(5, 5),
            goals=[(0, 0), (0, 9), (9, 0), (9, 9)],
            obstacles=[(4, 4), (4, 5), (5, 4), (5, 6), (6, 5)],
            goal_reward=15.0,
            step_penalty=-0.02,
            obstacle_penalty=-1.5,
            max_steps=200
        )
    }
    
    # Training parameters
    num_training_episodes = 500
    
    # Train agents for each environment
    results_summary = {}
    
    for env_name, config in environments.items():
        print(f"\n\n{'#'*60}")
        print(f"# Environment: {env_name}")
        print(f"{'#'*60}")
        
        # Create environment
        env = ConfigurableGridWorld(config=config)
        
        # Create agent
        agent = DQNAgent(
            state_size=2,
            action_size=4,
            learning_rate=0.001,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=64,
            target_update_freq=10
        )
        
        # Train
        train_stats = train_agent(env, agent, num_training_episodes, save_dir, env_name)
        
        # Evaluate
        print("\nEvaluating agent...")
        eval_results = evaluate_agent(env, agent, num_episodes=20)
        print(f"  Avg Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"  Avg Steps: {eval_results['avg_length']:.1f} ± {eval_results['std_length']:.1f}")
        print(f"  Success Rate: {eval_results['success_rate']:.2%}")
        
        results_summary[env_name] = eval_results
        
        # Visualize
        env_render = ConfigurableGridWorld(config=config, render_mode="human")
        agent_path = os.path.join(save_dir, f'agent_{env_name}.pth')
        agent.load(agent_path)
        visualize_agent(env_render, agent, save_dir, env_name, num_episodes=3)
        
        env.close()
        env_render.close()
    
    # Save summary
    summary_path = os.path.join(save_dir, 'results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"\n✓ Results summary saved to {summary_path}")
    
    # Plot results
    plot_training_results(save_dir)
    
    print(f"\n{'='*60}")
    print("Training and Analysis Complete!")
    print(f"All results saved in: {save_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()