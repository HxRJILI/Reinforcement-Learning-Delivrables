import numpy as np
import matplotlib.pyplot as plt
from configurable_gridworld import ConfigurableGridWorld, GridWorldConfig
from qlearning_agent import QLearningAgent, train_agent, evaluate_agent
import pickle
import os
from typing import Dict, List, Tuple
from datetime import datetime


def create_scaled_config(grid_size: int) -> GridWorldConfig:
    """
    Create a grid world configuration that scales appropriately with grid size.
    
    Args:
        grid_size: Size of the grid (NxN)
    
    Returns:
        GridWorldConfig with scaled parameters
    """
    # Start position: always bottom-left
    start_pos = (0, 0)
    
    # Goal position: always top-right corner
    goals = [(grid_size - 1, grid_size - 1)]
    
    # Scale obstacles based on grid size
    # Add obstacles proportionally to grid area
    obstacles = []
    num_obstacles = max(1, int(grid_size * 0.4))  # ~40% of grid size
    
    # Create a wall-like obstacle pattern
    if grid_size >= 5:
        mid = grid_size // 2
        for i in range(min(num_obstacles, grid_size - 2)):
            # Create vertical wall with gaps
            if i % 3 != 0:  # Leave some gaps
                obstacles.append((mid, i + 1))
    
    # Reward structure
    goal_reward = 1.0
    step_penalty = -0.01
    obstacle_penalty = -0.5
    
    # Max steps scales with grid size (allowing diagonal-like paths)
    max_steps = grid_size * grid_size * 2
    
    return GridWorldConfig(
        grid_size=grid_size,
        start_pos=start_pos,
        goals=goals,
        obstacles=obstacles,
        goal_reward=goal_reward,
        step_penalty=step_penalty,
        obstacle_penalty=obstacle_penalty,
        max_steps=max_steps,
        action_success_prob=1.0,
        slip_prob=0.0
    )


def train_for_grid_size(
    grid_size: int,
    num_episodes: int,
    learning_rate: float = 0.1,
    discount_factor: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    eval_interval: int = 50,
    eval_episodes: int = 10,
    seed: int = 42,
    save_dir: str = "trained_agents"
) -> Dict:
    """
    Train a Q-learning agent on a specific grid size and return metrics.
    
    Args:
        grid_size: Size of the grid
        num_episodes: Number of training episodes
        learning_rate: Learning rate for Q-learning
        discount_factor: Discount factor (gamma)
        epsilon: Initial exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Epsilon decay rate
        eval_interval: Evaluate every N episodes
        eval_episodes: Number of episodes for evaluation
        seed: Random seed
        save_dir: Directory to save trained agents
    
    Returns:
        Dictionary containing training metrics and agent filepath
    """
    print(f"\n{'='*60}")
    print(f"Training on Grid Size: {grid_size}x{grid_size}")
    print(f"{'='*60}")
    
    # Create environment configuration
    config = create_scaled_config(grid_size)
    env = ConfigurableGridWorld(config=config, render_mode=None)
    
    # Create agent
    agent = QLearningAgent(
        num_actions=4,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay
    )
    
    # Train agent
    metrics = train_agent(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        verbose=True,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes
    )
    
    # Final evaluation
    final_reward, final_steps = evaluate_agent(env, agent, num_episodes=100)
    
    print(f"\nFinal Performance:")
    print(f"  Average Reward: {final_reward:.3f}")
    print(f"  Average Steps: {final_steps:.1f}")
    print(f"  Q-table Size: {len(agent.q_table)}")
    
    # Save agent
    os.makedirs(save_dir, exist_ok=True)
    agent_filepath = os.path.join(save_dir, f'agent_grid{grid_size}x{grid_size}.pkl')
    agent.save(agent_filepath)
    
    # Save config alongside agent
    config_filepath = os.path.join(save_dir, f'config_grid{grid_size}x{grid_size}.pkl')
    with open(config_filepath, 'wb') as f:
        pickle.dump(config, f)
    print(f"Configuration saved to {config_filepath}")
    
    # Add metadata
    metrics['grid_size'] = grid_size
    metrics['final_reward'] = final_reward
    metrics['final_steps'] = final_steps
    metrics['q_table_size'] = len(agent.q_table)
    metrics['config'] = config
    metrics['agent_filepath'] = agent_filepath
    metrics['config_filepath'] = config_filepath
    
    env.close()
    
    return metrics


def plot_convergence_curves(
    all_metrics: Dict[int, Dict],
    save_dir: str = "results"
):
    """
    Plot convergence curves for different grid sizes.
    
    Args:
        all_metrics: Dictionary mapping grid_size to metrics
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    grid_sizes = sorted(all_metrics.keys())
    
    # Create color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(grid_sizes)))
    
    # Figure 1: Training Rewards over Episodes
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for idx, grid_size in enumerate(grid_sizes):
        metrics = all_metrics[grid_size]
        rewards = metrics['training_rewards']
        
        # Smooth the curve using moving average
        window = 50
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            episodes = np.arange(window-1, len(rewards))
            ax.plot(episodes, smoothed, label=f'{grid_size}x{grid_size}', 
                   color=colors[idx], linewidth=2)
        else:
            ax.plot(rewards, label=f'{grid_size}x{grid_size}', 
                   color=colors[idx], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward (50-episode moving average)', fontsize=12)
    ax.set_title('Training Convergence: Reward vs Episode for Different Grid Sizes', 
                fontsize=14, fontweight='bold')
    ax.legend(title='Grid Size', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_rewards.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(save_dir, 'convergence_rewards.png')}")
    
    # Figure 2: Evaluation Rewards
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for idx, grid_size in enumerate(grid_sizes):
        metrics = all_metrics[grid_size]
        eval_episodes = metrics['eval_episodes']
        eval_rewards = metrics['eval_rewards']
        
        ax.plot(eval_episodes, eval_rewards, marker='o', label=f'{grid_size}x{grid_size}',
               color=colors[idx], linewidth=2, markersize=6)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Evaluation Reward', fontsize=12)
    ax.set_title('Evaluation Performance vs Episode for Different Grid Sizes',
                fontsize=14, fontweight='bold')
    ax.legend(title='Grid Size', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_eval_rewards.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(save_dir, 'convergence_eval_rewards.png')}")
    
    # Figure 3: Training Steps over Episodes
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for idx, grid_size in enumerate(grid_sizes):
        metrics = all_metrics[grid_size]
        steps = metrics['training_steps']
        
        # Smooth the curve
        window = 50
        if len(steps) >= window:
            smoothed = np.convolve(steps, np.ones(window)/window, mode='valid')
            episodes = np.arange(window-1, len(steps))
            ax.plot(episodes, smoothed, label=f'{grid_size}x{grid_size}',
                   color=colors[idx], linewidth=2)
        else:
            ax.plot(steps, label=f'{grid_size}x{grid_size}',
                   color=colors[idx], linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Steps to Goal (50-episode moving average)', fontsize=12)
    ax.set_title('Training Convergence: Steps vs Episode for Different Grid Sizes',
                fontsize=14, fontweight='bold')
    ax.legend(title='Grid Size', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_steps.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(save_dir, 'convergence_steps.png')}")
    
    # Figure 4: Q-table Size Growth
    fig, ax = plt.subplots(figsize=(12, 6))
    
    q_table_sizes = [all_metrics[gs]['q_table_size'] for gs in grid_sizes]
    state_space_sizes = [gs * gs for gs in grid_sizes]
    
    ax.plot(grid_sizes, q_table_sizes, marker='o', linewidth=2, 
           markersize=8, label='Visited States', color='blue')
    ax.plot(grid_sizes, state_space_sizes, marker='s', linewidth=2, 
           markersize=8, label='Total State Space', color='red', linestyle='--')
    
    ax.set_xlabel('Grid Size', fontsize=12)
    ax.set_ylabel('Number of States', fontsize=12)
    ax.set_title('State Space Exploration vs Grid Size',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'state_space_growth.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(save_dir, 'state_space_growth.png')}")
    
    # Figure 5: Final Performance Summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    final_rewards = [all_metrics[gs]['final_reward'] for gs in grid_sizes]
    final_steps = [all_metrics[gs]['final_steps'] for gs in grid_sizes]
    
    ax1.bar(range(len(grid_sizes)), final_rewards, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(grid_sizes)))
    ax1.set_xticklabels([f'{gs}x{gs}' for gs in grid_sizes])
    ax1.set_xlabel('Grid Size', fontsize=12)
    ax1.set_ylabel('Final Average Reward', fontsize=12)
    ax1.set_title('Final Evaluation Reward by Grid Size', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(range(len(grid_sizes)), final_steps, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(grid_sizes)))
    ax2.set_xticklabels([f'{gs}x{gs}' for gs in grid_sizes])
    ax2.set_xlabel('Grid Size', fontsize=12)
    ax2.set_ylabel('Final Average Steps', fontsize=12)
    ax2.set_title('Final Steps to Goal by Grid Size', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_performance.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(save_dir, 'final_performance.png')}")
    
    plt.show()


def save_results(all_metrics: Dict[int, Dict], save_dir: str = "results"):
    """Save all metrics to a pickle file."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(save_dir, f'convergence_metrics_{timestamp}.pkl')
    
    with open(filepath, 'wb') as f:
        pickle.dump(all_metrics, f)
    
    print(f"\nMetrics saved to: {filepath}")


def load_trained_agent(grid_size: int, agents_dir: str = "trained_agents") -> Tuple[QLearningAgent, GridWorldConfig]:
    """
    Load a trained agent and its configuration.
    
    Args:
        grid_size: Size of the grid for which to load the agent
        agents_dir: Directory where agents are saved
    
    Returns:
        Tuple of (agent, config)
    """
    agent_filepath = os.path.join(agents_dir, f'agent_grid{grid_size}x{grid_size}.pkl')
    config_filepath = os.path.join(agents_dir, f'config_grid{grid_size}x{grid_size}.pkl')
    
    if not os.path.exists(agent_filepath):
        raise FileNotFoundError(f"Agent file not found: {agent_filepath}")
    if not os.path.exists(config_filepath):
        raise FileNotFoundError(f"Config file not found: {config_filepath}")
    
    # Load agent
    agent = QLearningAgent(num_actions=4)
    agent.load(agent_filepath)
    
    # Load config
    with open(config_filepath, 'rb') as f:
        config = pickle.load(f)
    
    return agent, config


def run_inference(
    grid_size: int,
    num_episodes: int = 10,
    agents_dir: str = "trained_agents",
    render: bool = True,
    render_delay: float = 0.3
):
    """
    Run inference using a trained agent.
    
    Args:
        grid_size: Size of the grid
        num_episodes: Number of episodes to run
        agents_dir: Directory where agents are saved
        render: Whether to render the environment
        render_delay: Delay between steps for visualization (seconds)
    """
    import time
    
    print(f"\n{'='*60}")
    print(f"RUNNING INFERENCE ON {grid_size}x{grid_size} GRID")
    print(f"{'='*60}")
    
    # Load trained agent and config
    agent, config = load_trained_agent(grid_size, agents_dir)
    
    print(f"\nAgent Statistics:")
    stats = agent.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create environment
    render_mode = "human" if render else None
    env = ConfigurableGridWorld(config=config, render_mode=render_mode)
    
    # Run episodes
    episode_rewards = []
    episode_steps = []
    success_count = 0
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        step_count = 0
        terminated = False
        truncated = False
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not (terminated or truncated):
            # Get action from trained agent (greedy policy)
            action = agent.get_action(state, training=False)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            state = next_state
            
            if render:
                time.sleep(render_delay)
        
        episode_rewards.append(episode_reward)
        episode_steps.append(step_count)
        
        if terminated:
            success_count += 1
            print(f"  ✓ Success! Reward: {episode_reward:.3f}, Steps: {step_count}")
        else:
            print(f"  ✗ Timeout. Reward: {episode_reward:.3f}, Steps: {step_count}")
    
    env.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Average Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Average Steps: {np.mean(episode_steps):.1f} ± {np.std(episode_steps):.1f}")
    print(f"{'='*60}")


def compare_all_agents(agents_dir: str = "trained_agents", num_episodes: int = 50):
    """
    Load and compare all trained agents.
    
    Args:
        agents_dir: Directory where agents are saved
        num_episodes: Number of episodes for evaluation
    """
    import glob
    
    print(f"\n{'='*70}")
    print(f"COMPARING ALL TRAINED AGENTS")
    print(f"{'='*70}")
    
    # Find all agent files
    agent_files = sorted(glob.glob(os.path.join(agents_dir, "agent_grid*.pkl")))
    
    if not agent_files:
        print(f"No trained agents found in {agents_dir}")
        return
    
    results = []
    
    for agent_file in agent_files:
        # Extract grid size from filename
        filename = os.path.basename(agent_file)
        grid_size = int(filename.split('grid')[1].split('x')[0])
        
        # Load agent and config
        agent, config = load_trained_agent(grid_size, agents_dir)
        
        # Create environment
        env = ConfigurableGridWorld(config=config, render_mode=None)
        
        # Evaluate
        avg_reward, avg_steps = evaluate_agent(env, agent, num_episodes=num_episodes)
        
        # Count successes
        success_count = 0
        for _ in range(num_episodes):
            state, _ = env.reset()
            terminated = False
            truncated = False
            steps = 0
            
            while not (terminated or truncated) and steps < config.max_steps:
                action = agent.get_action(state, training=False)
                state, _, terminated, truncated, _ = env.step(action)
                steps += 1
            
            if terminated:
                success_count += 1
        
        env.close()
        
        results.append({
            'grid_size': grid_size,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'success_rate': 100 * success_count / num_episodes,
            'q_table_size': len(agent.q_table)
        })
    
    # Print comparison table
    print(f"\n{'Grid Size':<12} {'Avg Reward':<15} {'Avg Steps':<15} {'Success Rate':<15} {'Q-Table Size':<15}")
    print("-"*70)
    for result in results:
        print(f"{result['grid_size']}x{result['grid_size']:<8} "
              f"{result['avg_reward']:<15.3f} "
              f"{result['avg_steps']:<15.1f} "
              f"{result['success_rate']:<14.1f}% "
              f"{result['q_table_size']:<15}")
    print("="*70)


def main():
    """Main function to run convergence analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Grid World Convergence Analysis')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'inference', 'compare'],
                       help='Mode: train new agents, run inference, or compare all agents')
    parser.add_argument('--grid-size', type=int, default=None,
                       help='Grid size for inference mode')
    parser.add_argument('--num-episodes', type=int, default=None,
                       help='Number of episodes (for training or inference)')
    parser.add_argument('--agents-dir', type=str, default='trained_agents',
                       help='Directory for saving/loading agents')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering in inference mode')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Training mode
        grid_sizes = [5, 7, 10, 12, 15]  # Different grid sizes to test
        num_episodes = args.num_episodes if args.num_episodes else 2000
        eval_interval = 50   # Evaluate every N episodes
        
        # Hyperparameters
        learning_rate = 0.1
        discount_factor = 0.99
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_decay = 0.995
        
        print("="*70)
        print("GRID SIZE CONVERGENCE ANALYSIS - TRAINING MODE")
        print("="*70)
        print(f"\nGrid Sizes: {grid_sizes}")
        print(f"Episodes per Grid: {num_episodes}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Discount Factor: {discount_factor}")
        print(f"Epsilon Decay: {epsilon_decay}")
        print(f"Agents will be saved to: {args.agents_dir}/")
        print("="*70)
        
        # Train agents for each grid size
        all_metrics = {}
        
        for grid_size in grid_sizes:
            metrics = train_for_grid_size(
                grid_size=grid_size,
                num_episodes=num_episodes,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay,
                eval_interval=eval_interval,
                eval_episodes=10,
                seed=42,
                save_dir=args.agents_dir
            )
            all_metrics[grid_size] = metrics
        
        # Generate plots
        print("\n" + "="*70)
        print("GENERATING CONVERGENCE PLOTS")
        print("="*70)
        plot_convergence_curves(all_metrics)
        
        # Save results
        save_results(all_metrics)
        
        # Print summary
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"{'Grid Size':<12} {'Final Reward':<15} {'Final Steps':<15} {'Q-Table Size':<15} {'Agent File':<30}")
        print("-"*100)
        for grid_size in sorted(all_metrics.keys()):
            metrics = all_metrics[grid_size]
            agent_file = os.path.basename(metrics['agent_filepath'])
            print(f"{grid_size}x{grid_size:<8} "
                  f"{metrics['final_reward']:<15.3f} "
                  f"{metrics['final_steps']:<15.1f} "
                  f"{metrics['q_table_size']:<15} "
                  f"{agent_file:<30}")
        print("="*100)
        print(f"\n✓ All agents saved to: {args.agents_dir}/")
        print(f"✓ Use '--mode inference --grid-size <size>' to test a trained agent")
        print(f"✓ Use '--mode compare' to compare all trained agents")
    
    elif args.mode == 'inference':
        # Inference mode
        if args.grid_size is None:
            print("Error: --grid-size required for inference mode")
            print("Example: python grid_convergence_analysis.py --mode inference --grid-size 10")
            return
        
        num_episodes = args.num_episodes if args.num_episodes else 10
        run_inference(
            grid_size=args.grid_size,
            num_episodes=num_episodes,
            agents_dir=args.agents_dir,
            render=not args.no_render,
            render_delay=0.3
        )
    
    elif args.mode == 'compare':
        # Compare all agents
        num_episodes = args.num_episodes if args.num_episodes else 50
        compare_all_agents(agents_dir=args.agents_dir, num_episodes=num_episodes)


if __name__ == "__main__":
    main()