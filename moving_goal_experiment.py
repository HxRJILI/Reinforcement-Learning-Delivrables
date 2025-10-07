import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import pickle
import os
from typing import List, Tuple, Dict
from configurable_gridworld import ConfigurableGridWorld, GridWorldConfig
from qlearning_agent import QLearningAgent, evaluate_agent


class MovingGoalExperiment:
    """
    Experiment to demonstrate that Q-learning fails when goals change between episodes.
    
    This compares two scenarios:
    1. Fixed Goal: Goal remains constant across all episodes (baseline)
    2. Moving Goal: Goal changes randomly each episode
    """
    
    def __init__(
        self,
        grid_size: int = 8,
        start_pos: Tuple[int, int] = (0, 0),
        obstacles: List[Tuple[int, int]] = None,
        possible_goals: List[Tuple[int, int]] = None,
        num_episodes: int = 1000,
        max_steps_per_episode: int = 100
    ):
        """
        Initialize the experiment.
        
        Args:
            grid_size: Size of the grid
            start_pos: Starting position (same for all episodes)
            obstacles: List of obstacle positions
            possible_goals: List of possible goal positions for moving goal scenario
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
        """
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.obstacles = obstacles if obstacles else []
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        
        # Define possible goals if not provided
        if possible_goals is None:
            # Create goals at corners and edges
            self.possible_goals = [
                (grid_size-1, grid_size-1),  # Bottom-right
                (grid_size-1, 0),            # Top-right
                (0, grid_size-1),            # Bottom-left
                (grid_size//2, grid_size-1), # Bottom-center
                (grid_size-1, grid_size//2), # Right-center
            ]
        else:
            self.possible_goals = possible_goals
        
        # Remove any goals that overlap with obstacles or start
        self.possible_goals = [
            g for g in self.possible_goals 
            if g not in self.obstacles and g != start_pos
        ]
        
        self.results_dir = "Moving_Goal_Results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Experiment initialized:")
        print(f"  Grid size: {grid_size}x{grid_size}")
        print(f"  Start position: {start_pos}")
        print(f"  Obstacles: {len(self.obstacles)}")
        print(f"  Possible goals: {self.possible_goals}")
        print(f"  Episodes: {num_episodes}")
    
    def run_fixed_goal_experiment(self, fixed_goal: Tuple[int, int]) -> Dict:
        """
        Train agent with a FIXED goal (baseline scenario).
        
        Args:
            fixed_goal: The goal position that remains constant
            
        Returns:
            Dictionary with training metrics and agent
        """
        print("\n" + "="*60)
        print("FIXED GOAL EXPERIMENT (Baseline)")
        print("="*60)
        print(f"Goal position: {fixed_goal}")
        
        # Create environment with fixed goal
        config = GridWorldConfig(
            grid_size=self.grid_size,
            start_pos=self.start_pos,
            goals=[fixed_goal],
            obstacles=self.obstacles,
            goal_reward=10.0,
            step_penalty=-0.1,
            obstacle_penalty=-1.0,
            max_steps=self.max_steps_per_episode
        )
        
        env = ConfigurableGridWorld(config=config, render_mode=None)
        
        # Create agent
        agent = QLearningAgent(
            num_actions=4,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995
        )
        
        # Training metrics
        episode_rewards = []
        episode_steps = []
        episode_success = []  # Did the agent reach the goal?
        q_table_sizes = []
        epsilons = []
        
        for episode in range(self.num_episodes):
            state, info = env.reset()
            episode_reward = 0
            episode_step = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated) and episode_step < self.max_steps_per_episode:
                action = agent.get_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                agent.update(state, action, reward, next_state, terminated)
                
                episode_reward += reward
                episode_step += 1
                state = next_state
            
            agent.end_episode()
            
            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)
            episode_success.append(1 if terminated else 0)
            q_table_sizes.append(len(agent.q_table))
            epsilons.append(agent.epsilon)
            
            if (episode + 1) % 100 == 0:
                recent_success_rate = np.mean(episode_success[-100:]) * 100
                recent_avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode+1}/{self.num_episodes} - "
                      f"Success Rate: {recent_success_rate:.1f}%, "
                      f"Avg Reward: {recent_avg_reward:.2f}, "
                      f"Q-table size: {len(agent.q_table)}, "
                      f"Epsilon: {agent.epsilon:.3f}")
        
        env.close()
        
        return {
            'agent': agent,
            'rewards': episode_rewards,
            'steps': episode_steps,
            'success': episode_success,
            'q_table_sizes': q_table_sizes,
            'epsilons': epsilons,
            'config': config
        }
    
    def run_moving_goal_experiment(self, seed: int = 42) -> Dict:
        """
        Train agent with MOVING goals (changes each episode).
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with training metrics and agent
        """
        print("\n" + "="*60)
        print("MOVING GOAL EXPERIMENT")
        print("="*60)
        print(f"Goal changes each episode from: {self.possible_goals}")
        
        np.random.seed(seed)
        
        # Create base environment (we'll change the goal each episode)
        base_config = GridWorldConfig(
            grid_size=self.grid_size,
            start_pos=self.start_pos,
            goals=[self.possible_goals[0]],  # Initial goal
            obstacles=self.obstacles,
            goal_reward=10.0,
            step_penalty=-0.1,
            obstacle_penalty=-1.0,
            max_steps=self.max_steps_per_episode
        )
        
        env = ConfigurableGridWorld(config=base_config, render_mode=None)
        
        # Create agent
        agent = QLearningAgent(
            num_actions=4,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995
        )
        
        # Training metrics
        episode_rewards = []
        episode_steps = []
        episode_success = []
        q_table_sizes = []
        epsilons = []
        goal_history = []  # Track which goal was used each episode
        
        for episode in range(self.num_episodes):
            # Randomly select a goal for this episode
            current_goal = self.possible_goals[np.random.randint(len(self.possible_goals))]
            goal_history.append(current_goal)
            
            # Update environment configuration with new goal
            new_config = GridWorldConfig(
                grid_size=self.grid_size,
                start_pos=self.start_pos,
                goals=[current_goal],
                obstacles=self.obstacles,
                goal_reward=10.0,
                step_penalty=-0.1,
                obstacle_penalty=-1.0,
                max_steps=self.max_steps_per_episode
            )
            
            state, info = env.reset(options={'config': new_config})
            episode_reward = 0
            episode_step = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated) and episode_step < self.max_steps_per_episode:
                action = agent.get_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                agent.update(state, action, reward, next_state, terminated)
                
                episode_reward += reward
                episode_step += 1
                state = next_state
            
            agent.end_episode()
            
            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)
            episode_success.append(1 if terminated else 0)
            q_table_sizes.append(len(agent.q_table))
            epsilons.append(agent.epsilon)
            
            if (episode + 1) % 100 == 0:
                recent_success_rate = np.mean(episode_success[-100:]) * 100
                recent_avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode+1}/{self.num_episodes} - "
                      f"Success Rate: {recent_success_rate:.1f}%, "
                      f"Avg Reward: {recent_avg_reward:.2f}, "
                      f"Q-table size: {len(agent.q_table)}, "
                      f"Epsilon: {agent.epsilon:.3f}")
        
        env.close()
        
        return {
            'agent': agent,
            'rewards': episode_rewards,
            'steps': episode_steps,
            'success': episode_success,
            'q_table_sizes': q_table_sizes,
            'epsilons': epsilons,
            'goal_history': goal_history,
            'config': base_config
        }
    
    def visualize_comparison(self, fixed_results: Dict, moving_results: Dict):
        """
        Create comprehensive visualization comparing fixed vs moving goal scenarios.
        """
        print("\nGenerating visualizations...")
        
        # Calculate moving averages for smoothing
        window = 50
        
        def moving_average(data, window):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Rewards over time
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(moving_average(fixed_results['rewards'], window), 
                label='Fixed Goal', color='green', linewidth=2, alpha=0.8)
        ax1.plot(moving_average(moving_results['rewards'], window), 
                label='Moving Goal', color='red', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Episode', fontsize=11)
        ax1.set_ylabel('Reward (Moving Avg)', fontsize=11)
        ax1.set_title('Learning Progress: Reward Over Time', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Success rate over time
        ax2 = fig.add_subplot(gs[0, 2])
        fixed_success_rate = [np.mean(fixed_results['success'][max(0, i-100):i+1]) * 100 
                              for i in range(len(fixed_results['success']))]
        moving_success_rate = [np.mean(moving_results['success'][max(0, i-100):i+1]) * 100 
                               for i in range(len(moving_results['success']))]
        
        ax2.plot(fixed_success_rate, color='green', linewidth=2, alpha=0.8, label='Fixed Goal')
        ax2.plot(moving_success_rate, color='red', linewidth=2, alpha=0.8, label='Moving Goal')
        ax2.set_xlabel('Episode', fontsize=11)
        ax2.set_ylabel('Success Rate (%)', fontsize=11)
        ax2.set_title('Success Rate\n(100-ep moving avg)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])
        
        # 3. Steps to goal over time
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(moving_average(fixed_results['steps'], window), 
                color='green', linewidth=2, alpha=0.8, label='Fixed Goal')
        ax3.plot(moving_average(moving_results['steps'], window), 
                color='red', linewidth=2, alpha=0.8, label='Moving Goal')
        ax3.set_xlabel('Episode', fontsize=11)
        ax3.set_ylabel('Steps (Moving Avg)', fontsize=11)
        ax3.set_title('Efficiency: Steps to Goal Over Time', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. Q-table size growth
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(fixed_results['q_table_sizes'], color='green', linewidth=2, alpha=0.8, label='Fixed Goal')
        ax4.plot(moving_results['q_table_sizes'], color='red', linewidth=2, alpha=0.8, label='Moving Goal')
        ax4.set_xlabel('Episode', fontsize=11)
        ax4.set_ylabel('Q-table Size', fontsize=11)
        ax4.set_title('State Space\nExploration', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. Final performance comparison (bar chart)
        ax5 = fig.add_subplot(gs[2, 0])
        categories = ['Success\nRate (%)', 'Avg\nReward', 'Avg\nSteps']
        
        # Calculate final performance (last 100 episodes)
        fixed_final = [
            np.mean(fixed_results['success'][-100:]) * 100,
            np.mean(fixed_results['rewards'][-100:]),
            np.mean(fixed_results['steps'][-100:])
        ]
        moving_final = [
            np.mean(moving_results['success'][-100:]) * 100,
            np.mean(moving_results['rewards'][-100:]),
            np.mean(moving_results['steps'][-100:])
        ]
        
        # Normalize for visualization (except success rate)
        fixed_normalized = [fixed_final[0], fixed_final[1], fixed_final[2]]
        moving_normalized = [moving_final[0], moving_final[1], moving_final[2]]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, fixed_normalized, width, label='Fixed Goal', 
                       color='green', alpha=0.7)
        bars2 = ax5.bar(x + width/2, moving_normalized, width, label='Moving Goal', 
                       color='red', alpha=0.7)
        
        ax5.set_ylabel('Value', fontsize=11)
        ax5.set_title('Final Performance\n(Last 100 Episodes)', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(categories, fontsize=9)
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 6. Epsilon decay comparison
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(fixed_results['epsilons'], color='green', linewidth=2, alpha=0.8, label='Fixed Goal')
        ax6.plot(moving_results['epsilons'], color='red', linewidth=2, alpha=0.8, label='Moving Goal')
        ax6.set_xlabel('Episode', fontsize=11)
        ax6.set_ylabel('Epsilon', fontsize=11)
        ax6.set_title('Exploration Rate\n(Epsilon)', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        # 7. Reward distribution (last 200 episodes)
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.hist(fixed_results['rewards'][-200:], bins=30, alpha=0.6, 
                color='green', label='Fixed Goal', density=True)
        ax7.hist(moving_results['rewards'][-200:], bins=30, alpha=0.6, 
                color='red', label='Moving Goal', density=True)
        ax7.set_xlabel('Reward', fontsize=11)
        ax7.set_ylabel('Density', fontsize=11)
        ax7.set_title('Reward Distribution\n(Last 200 Episodes)', fontsize=12, fontweight='bold')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Q-Learning Performance: Fixed Goal vs Moving Goal', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        filepath = os.path.join(self.results_dir, 'comprehensive_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.show()
    
    def visualize_q_value_heatmaps(self, fixed_results: Dict, moving_results: Dict):
        """
        Visualize Q-value heatmaps showing learned values for different goals.
        """
        print("\nGenerating Q-value heatmaps...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Fixed goal Q-values
        fixed_agent = fixed_results['agent']
        fixed_config = fixed_results['config']
        fixed_goal = fixed_config.goals[0]
        
        # Moving goal Q-values
        moving_agent = moving_results['agent']
        
        # Create value heatmap for fixed goal agent
        fixed_values = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = np.array([i, j])
                fixed_values[i, j] = fixed_agent.get_value(state)
        
        # Create value heatmap for moving goal agent
        moving_values = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = np.array([i, j])
                moving_values[i, j] = moving_agent.get_value(state)
        
        # Plot fixed goal heatmap
        im1 = axes[0].imshow(fixed_values, cmap='RdYlGn', origin='upper')
        axes[0].set_title(f'Fixed Goal Agent\nState Values V(s)\nGoal: {fixed_goal}', 
                         fontsize=13, fontweight='bold')
        
        # Add goal marker
        axes[0].plot(fixed_goal[1], fixed_goal[0], 'r*', markersize=20, 
                    markeredgecolor='black', markeredgewidth=2)
        
        # Add obstacles
        for obs in self.obstacles:
            axes[0].add_patch(patches.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1, 
                                               fill=True, color='black', alpha=0.5))
        
        # Add start position
        axes[0].plot(self.start_pos[1], self.start_pos[0], 'go', markersize=15,
                    markeredgecolor='black', markeredgewidth=2)
        
        axes[0].set_xlabel('Y Position', fontsize=11)
        axes[0].set_ylabel('X Position', fontsize=11)
        plt.colorbar(im1, ax=axes[0], label='State Value')
        axes[0].grid(True, alpha=0.3)
        
        # Plot moving goal heatmap
        im2 = axes[1].imshow(moving_values, cmap='RdYlGn', origin='upper')
        axes[1].set_title('Moving Goal Agent\nState Values V(s)\n(Confused/No Clear Goal)', 
                         fontsize=13, fontweight='bold')
        
        # Add all possible goals
        for goal in self.possible_goals:
            axes[1].plot(goal[1], goal[0], 'r*', markersize=15, 
                        markeredgecolor='black', markeredgewidth=1.5, alpha=0.7)
        
        # Add obstacles
        for obs in self.obstacles:
            axes[1].add_patch(patches.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1, 
                                               fill=True, color='black', alpha=0.5))
        
        # Add start position
        axes[1].plot(self.start_pos[1], self.start_pos[0], 'go', markersize=15,
                    markeredgecolor='black', markeredgewidth=2)
        
        axes[1].set_xlabel('Y Position', fontsize=11)
        axes[1].set_ylabel('X Position', fontsize=11)
        plt.colorbar(im2, ax=axes[1], label='State Value')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.results_dir, 'q_value_heatmaps.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.show()
    
    def generate_summary_report(self, fixed_results: Dict, moving_results: Dict):
        """
        Generate a text summary report of the experiment.
        """
        print("\nGenerating summary report...")
        
        # Calculate statistics
        fixed_final_success = np.mean(fixed_results['success'][-100:]) * 100
        moving_final_success = np.mean(moving_results['success'][-100:]) * 100
        
        fixed_final_reward = np.mean(fixed_results['rewards'][-100:])
        moving_final_reward = np.mean(moving_results['rewards'][-100:])
        
        fixed_final_steps = np.mean(fixed_results['steps'][-100:])
        moving_final_steps = np.mean(moving_results['steps'][-100:])
        
        fixed_q_size = len(fixed_results['agent'].q_table)
        moving_q_size = len(moving_results['agent'].q_table)
        
        report = f"""
{'='*70}
        Q-LEARNING WITH MOVING GOALS: EXPERIMENTAL REPORT
{'='*70}

EXPERIMENT SETUP:
-----------------
Grid Size:              {self.grid_size}x{self.grid_size}
Start Position:         {self.start_pos}
Number of Episodes:     {self.num_episodes}
Max Steps per Episode:  {self.max_steps_per_episode}
Number of Obstacles:    {len(self.obstacles)}
Possible Goals:         {self.possible_goals}

AGENT CONFIGURATION:
--------------------
Learning Rate:          0.1
Discount Factor:        0.95
Initial Epsilon:        1.0
Min Epsilon:            0.01
Epsilon Decay:          0.995

RESULTS (Last 100 Episodes):
=============================

FIXED GOAL (Baseline):
----------------------
Success Rate:           {fixed_final_success:.2f}%
Average Reward:         {fixed_final_reward:.2f}
Average Steps:          {fixed_final_steps:.2f}
Q-table Size:           {fixed_q_size} states
Goal Position:          {fixed_results['config'].goals[0]}

MOVING GOAL:
------------
Success Rate:           {moving_final_success:.2f}%
Average Reward:         {moving_final_reward:.2f}
Average Steps:          {moving_final_steps:.2f}
Q-table Size:           {moving_q_size} states

PERFORMANCE DEGRADATION:
------------------------
Success Rate Drop:      {fixed_final_success - moving_final_success:.2f}%
Reward Drop:            {fixed_final_reward - moving_final_reward:.2f}
Steps Increase:         {moving_final_steps - fixed_final_steps:.2f}

CONCLUSIONS:
============

1. LEARNING FAILURE WITH MOVING GOALS:
   The agent with moving goals achieved a success rate of {moving_final_success:.1f}%
   compared to {fixed_final_success:.1f}% with a fixed goal - a degradation of
   {fixed_final_success - moving_final_success:.1f} percentage points.

2. WHY THIS HAPPENS:
   - Q-learning assumes a stationary environment (MDP with fixed transitions/rewards)
   - When the goal changes, the optimal policy changes, making previously learned
     Q-values obsolete or misleading
   - The agent wastes time exploring states that lead to goals that are no longer active
   - Each episode with a different goal essentially "unlearns" progress from previous
     episodes with different goals

3. STATE SPACE EXPLORATION:
   The moving goal agent explored {moving_q_size} states vs {fixed_q_size} for
   fixed goal, showing {'more' if moving_q_size > fixed_q_size else 'less'} exploration
   but {'no benefit' if moving_final_success < fixed_final_success else 'some benefit'} in performance.

4. PRACTICAL IMPLICATIONS:
   This demonstrates that standard Q-learning is NOT suitable for:
   - Non-stationary environments
   - Multi-task learning without task identification
   - Scenarios where goals/objectives change over time
   
   Alternative approaches needed:
   - Goal-conditioned RL (HER, UVFA)
   - Meta-RL / Learning to learn
   - Context-aware policies
   - Hierarchical RL with goal abstraction

{'='*70}
Report generated: {os.path.basename(self.results_dir)}
{'='*70}
"""
        
        filepath = os.path.join(self.results_dir, 'experiment_report.txt')
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"\nSaved: {filepath}")
    
    def save_results(self, fixed_results: Dict, moving_results: Dict):
        """Save experimental results to pickle files."""
        print("\nSaving results...")
        
        # Remove agents before saving (they're large)
        fixed_save = {k: v for k, v in fixed_results.items() if k != 'agent'}
        moving_save = {k: v for k, v in moving_results.items() if k != 'agent'}
        
        # Save results
        with open(os.path.join(self.results_dir, 'fixed_goal_results.pkl'), 'wb') as f:
            pickle.dump(fixed_save, f)
        
        with open(os.path.join(self.results_dir, 'moving_goal_results.pkl'), 'wb') as f:
            pickle.dump(moving_save, f)
        
        # Save agents separately
        fixed_results['agent'].save(os.path.join(self.results_dir, 'fixed_goal_agent.pkl'))
        moving_results['agent'].save(os.path.join(self.results_dir, 'moving_goal_agent.pkl'))
        
        print(f"Results saved to {self.results_dir}/")
    
    def run_full_experiment(self):
        """Run the complete experiment with all visualizations."""
        print("\n" + "="*70)
        print("STARTING MOVING GOAL EXPERIMENT")
        print("="*70)
        
        # Select a fixed goal (e.g., bottom-right corner)
        fixed_goal = self.possible_goals[0]
        
        # Run fixed goal experiment
        fixed_results = self.run_fixed_goal_experiment(fixed_goal)
        
        # Run moving goal experiment
        moving_results = self.run_moving_goal_experiment(seed=42)
        
        # Generate visualizations
        self.visualize_comparison(fixed_results, moving_results)
        self.visualize_q_value_heatmaps(fixed_results, moving_results)
        
        # Generate report
        self.generate_summary_report(fixed_results, moving_results)
        
        # Save results
        self.save_results(fixed_results, moving_results)
        
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE!")
        print(f"All results saved to: {self.results_dir}/")
        print("="*70)


if __name__ == "__main__":
    # Configure the experiment
    experiment = MovingGoalExperiment(
        grid_size=8,
        start_pos=(0, 0),
        obstacles=[(3, 3), (3, 4), (4, 3), (4, 4), (2, 5), (5, 2), (6, 1)],
        possible_goals=[
            (7, 7),  # Bottom-right
            (7, 0),  # Top-right
            (0, 7),  # Bottom-left
            (7, 3),  # Right-middle
            (3, 7),  # Bottom-middle
        ],
        num_episodes=1000,
        max_steps_per_episode=100
    )
    
    # Run the full experiment
    experiment.run_full_experiment()
    
    print("\n" + "="*70)
    print("ADDITIONAL ANALYSIS")
    print("="*70)
    
    # Optional: Run a shorter experiment with different parameters
    print("\n\nRunning additional experiment with more goals...")
    experiment2 = MovingGoalExperiment(
        grid_size=10,
        start_pos=(0, 0),
        obstacles=[(4, 4), (4, 5), (5, 4), (5, 5), (2, 7), (7, 2), (8, 3)],
        possible_goals=[
            (9, 9),  # Bottom-right corner
            (9, 0),  # Top-right corner
            (0, 9),  # Bottom-left corner
            (9, 4),  # Right-middle
            (4, 9),  # Bottom-middle
            (5, 0),  # Top-middle
            (0, 5),  # Left-middle
        ],
        num_episodes=800,
        max_steps_per_episode=150
    )
    
    # This will create a second set of results in the same folder
    # IMPORTANT: Create the directory before running
    experiment2.results_dir = "Moving_Goal_Results/10x10_grid"
    os.makedirs(experiment2.results_dir, exist_ok=True)  # ADD THIS LINE
    experiment2.run_full_experiment()
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print("\nGenerated files in Moving_Goal_Results/:")
    print("  - comprehensive_comparison.png (main results)")
    print("  - q_value_heatmaps.png (learned value functions)")
    print("  - experiment_report.txt (detailed analysis)")
    print("  - fixed_goal_results.pkl (training data)")
    print("  - moving_goal_results.pkl (training data)")
    print("  - fixed_goal_agent.pkl (trained agent)")
    print("  - moving_goal_agent.pkl (trained agent)")
    print("\n  10x10_grid/ subfolder contains results from second experiment")
    print("\nKey findings will be in experiment_report.txt")
    print("="*70)