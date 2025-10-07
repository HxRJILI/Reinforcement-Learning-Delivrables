import numpy as np
from typing import Optional, Tuple, Dict, Any
import pickle
from collections import defaultdict
import os


class QLearningAgent:
    """
    Q-Learning agent for discrete state-action environments.
    
    Supports:
    - Epsilon-greedy exploration
    - Learning rate decay
    - Epsilon decay
    - State-action value function (Q-table)
    """
    
    def __init__(
        self,
        num_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate_decay: float = 1.0,
        learning_rate_min: float = 0.01
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            num_actions: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            learning_rate_decay: Decay rate for learning rate
            learning_rate_min: Minimum learning rate
        """
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_min = learning_rate_min
        
        # Q-table: dict mapping state to array of Q-values for each action
        self.q_table = defaultdict(lambda: np.zeros(num_actions))
        
        # Statistics
        self.training_episodes = 0
        self.total_steps = 0
    
    def state_to_key(self, state: np.ndarray) -> Tuple:
        """Convert state array to hashable tuple for Q-table lookup."""
        return tuple(state.flatten())
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions in a given state."""
        state_key = self.state_to_key(state)
        return self.q_table[state_key]
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, use greedy
        
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.num_actions)
        else:
            # Exploit: best action
            q_values = self.get_q_values(state)
            # Break ties randomly
            max_q = np.max(q_values)
            max_actions = np.where(q_values == max_q)[0]
            return np.random.choice(max_actions)
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool
    ):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            terminated: Whether episode terminated
        """
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Calculate target Q-value
        if terminated:
            target_q = reward
        else:
            next_max_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.discount_factor * next_max_q
        
        # Q-learning update
        self.q_table[state_key][action] = current_q + self.learning_rate * (target_q - current_q)
        
        self.total_steps += 1
    
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def decay_learning_rate(self):
        """Decay learning rate after each episode."""
        self.learning_rate = max(
            self.learning_rate_min,
            self.learning_rate * self.learning_rate_decay
        )
    
    def end_episode(self):
        """Called at the end of each episode to update parameters."""
        self.training_episodes += 1
        self.decay_epsilon()
        self.decay_learning_rate()
    
    def reset_epsilon(self, epsilon: Optional[float] = None):
        """Reset epsilon to initial or specified value."""
        if epsilon is not None:
            self.epsilon = epsilon
        else:
            self.epsilon = 1.0
    
    def reset_learning_rate(self, learning_rate: Optional[float] = None):
        """Reset learning rate to initial or specified value."""
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = self.initial_learning_rate
    
    def get_policy(self, state: np.ndarray) -> int:
        """Get the greedy policy action for a state."""
        q_values = self.get_q_values(state)
        return np.argmax(q_values)
    
    def get_value(self, state: np.ndarray) -> float:
        """Get the state value V(s) = max_a Q(s,a)."""
        q_values = self.get_q_values(state)
        return np.max(q_values)
    
    def save(self, filepath: str):
        """Save agent to file."""
        save_dict = {
            'q_table': dict(self.q_table),
            'num_actions': self.num_actions,
            'learning_rate': self.learning_rate,
            'initial_learning_rate': self.initial_learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'learning_rate_decay': self.learning_rate_decay,
            'learning_rate_min': self.learning_rate_min,
            'training_episodes': self.training_episodes,
            'total_steps': self.total_steps
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent from file."""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions), save_dict['q_table'])
        self.num_actions = save_dict['num_actions']
        self.learning_rate = save_dict['learning_rate']
        self.initial_learning_rate = save_dict['initial_learning_rate']
        self.discount_factor = save_dict['discount_factor']
        self.epsilon = save_dict['epsilon']
        self.epsilon_min = save_dict['epsilon_min']
        self.epsilon_decay = save_dict['epsilon_decay']
        self.learning_rate_decay = save_dict['learning_rate_decay']
        self.learning_rate_min = save_dict['learning_rate_min']
        self.training_episodes = save_dict['training_episodes']
        self.total_steps = save_dict['total_steps']
        print(f"Agent loaded from {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'training_episodes': self.training_episodes,
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'q_table_size': len(self.q_table),
            'avg_q_value': np.mean([np.mean(q_vals) for q_vals in self.q_table.values()]) if self.q_table else 0.0
        }


def train_agent(
    env,
    agent: QLearningAgent,
    num_episodes: int,
    max_steps_per_episode: int = 1000,
    verbose: bool = True,
    eval_interval: int = 100,
    eval_episodes: int = 10
) -> Dict[str, list]:
    """
    Train a Q-Learning agent on an environment.
    
    Args:
        env: Environment instance
        agent: QLearningAgent instance
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        verbose: Print progress
        eval_interval: Evaluate every N episodes
        eval_episodes: Number of episodes for evaluation
    
    Returns:
        Dictionary with training metrics
    """
    training_rewards = []
    training_steps = []
    eval_rewards = []
    eval_steps = []
    eval_episodes_list = []
    epsilons = []
    learning_rates = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and episode_steps < max_steps_per_episode:
            # Select and perform action
            action = agent.get_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Update Q-table
            agent.update(state, action, reward, next_state, terminated)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
        
        # End of episode
        agent.end_episode()
        training_rewards.append(episode_reward)
        training_steps.append(episode_steps)
        epsilons.append(agent.epsilon)
        learning_rates.append(agent.learning_rate)
        
        # Evaluation
        if (episode + 1) % eval_interval == 0:
            eval_reward, eval_step = evaluate_agent(env, agent, eval_episodes)
            eval_rewards.append(eval_reward)
            eval_steps.append(eval_step)
            eval_episodes_list.append(episode + 1)
            
            if verbose:
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Training - Reward: {np.mean(training_rewards[-eval_interval:]):.2f}, "
                      f"Steps: {np.mean(training_steps[-eval_interval:]):.1f}")
                print(f"  Eval     - Reward: {eval_reward:.2f}, Steps: {eval_step:.1f}")
                print(f"  Epsilon: {agent.epsilon:.3f}, LR: {agent.learning_rate:.4f}")
                print(f"  Q-table size: {len(agent.q_table)}")
    
    return {
        'training_rewards': training_rewards,
        'training_steps': training_steps,
        'eval_rewards': eval_rewards,
        'eval_steps': eval_steps,
        'eval_episodes': eval_episodes_list,
        'epsilons': epsilons,
        'learning_rates': learning_rates
    }


def evaluate_agent(
    env,
    agent: QLearningAgent,
    num_episodes: int = 10,
    max_steps_per_episode: int = 1000
) -> Tuple[float, float]:
    """
    Evaluate agent performance (greedy policy, no exploration).
    
    Args:
        env: Environment instance
        agent: QLearningAgent instance
        num_episodes: Number of evaluation episodes
        max_steps_per_episode: Maximum steps per episode
    
    Returns:
        Average reward and average steps
    """
    total_rewards = []
    total_steps = []
    
    for _ in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and episode_steps < max_steps_per_episode:
            # Greedy action (no exploration)
            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
        
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
    
    return np.mean(total_rewards), np.mean(total_steps)


def generate_agent_filename(env_config, base_dir="agents") -> str:
    """
    Generate a unique filename for saving the agent based on environment configuration.
    
    Args:
        env_config: Environment configuration object (e.g., GridWorldConfig)
        base_dir: Directory where agents are saved
        
    Returns:
        A file path string for saving the agent.
    """
    # Ensure save directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a descriptive name using key attributes
    name_parts = [
        f"grid{env_config.grid_size}",
        f"goals{len(env_config.goals)}",
        f"obst{len(env_config.obstacles)}",
        f"start{env_config.start_pos[0]}-{env_config.start_pos[1]}"
    ]
    
    filename = "_".join(name_parts) + ".pkl"
    return os.path.join(base_dir, filename)



# Example usage
if __name__ == "__main__":
    from configurable_gridworld import ConfigurableGridWorld, GridWorldConfig
    
    # Create environment
    config = GridWorldConfig(
        grid_size=5,
        start_pos=(0, 0),
        goals=[(4, 4)],
        obstacles=[(2, 2), (2, 3)],
        goal_reward=1.0,
        step_penalty=-0.01,
        obstacle_penalty=-0.5,
        max_steps=100
    )
    
    env = ConfigurableGridWorld(config=config, render_mode=None)
    
    # Create agent
    agent = QLearningAgent(
        num_actions=4,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # Train agent
    print("Training Q-Learning agent...")
    metrics = train_agent(
        env=env,
        agent=agent,
        num_episodes=1000,
        verbose=True,
        eval_interval=100,
        eval_episodes=10
    )
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_reward, final_steps = evaluate_agent(env, agent, num_episodes=100)
    print(f"Average Reward: {final_reward:.2f}")
    print(f"Average Steps: {final_steps:.1f}")
    
    # Save agent
    # Generate a name based on the environment
    agent_filepath = generate_agent_filename(config)

    # Save the agent
    agent.save(agent_filepath)

    env.close()