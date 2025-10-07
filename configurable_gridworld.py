import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class GridWorldConfig:
    """Configuration for the Grid World environment."""
    grid_size: int = 5
    start_pos: Tuple[int, int] = (0, 0)
    goals: List[Tuple[int, int]] = field(default_factory=lambda: [(4, 4)])
    obstacles: List[Tuple[int, int]] = field(default_factory=list)
    
    # Reward structure
    goal_reward: float = 1.0
    step_penalty: float = -0.01
    obstacle_penalty: float = -0.5
    
    # Dynamics parameters
    action_success_prob: float = 1.0  # Probability action succeeds
    slip_prob: float = 0.0  # Probability of slipping to adjacent direction
    
    # Episode parameters
    max_steps: int = 100
    
    def validate(self):
        """Validate configuration."""
        assert self.grid_size > 0, "Grid size must be positive"
        assert 0 <= self.start_pos[0] < self.grid_size, "Start position out of bounds"
        assert 0 <= self.start_pos[1] < self.grid_size, "Start position out of bounds"
        assert self.start_pos not in self.obstacles, "Start position cannot be an obstacle"
        assert self.start_pos not in self.goals, "Start position cannot be a goal"
        
        for goal in self.goals:
            assert 0 <= goal[0] < self.grid_size, f"Goal {goal} out of bounds"
            assert 0 <= goal[1] < self.grid_size, f"Goal {goal} out of bounds"
            assert goal not in self.obstacles, f"Goal {goal} cannot be an obstacle"
        
        for obs in self.obstacles:
            assert 0 <= obs[0] < self.grid_size, f"Obstacle {obs} out of bounds"
            assert 0 <= obs[1] < self.grid_size, f"Obstacle {obs} out of bounds"
        
        assert 0 <= self.action_success_prob <= 1, "Action success prob must be in [0, 1]"
        assert 0 <= self.slip_prob <= 1, "Slip probability must be in [0, 1]"


class ConfigurableGridWorld:
    """
    A highly configurable Grid World environment with Gymnasium philosophy.
    
    Features:
    - Multiple goals
    - Configurable obstacles
    - Stochastic dynamics (action noise, slipping)
    - Customizable rewards
    - Episode length limits
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, config: Optional[GridWorldConfig] = None, render_mode: Optional[str] = None):
        """
        Initialize the environment.
        
        Args:
            config: GridWorldConfig object with environment parameters
            render_mode: "human" for visualization, "rgb_array" for array output
        """
        self.config = config if config is not None else GridWorldConfig()
        self.config.validate()
        
        self.render_mode = render_mode
        
        # Action space: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space_n = 4
        
        # Observation space: position (x, y) of the agent
        self.observation_space_low = 0
        self.observation_space_high = self.config.grid_size - 1
        self.observation_space_shape = (2,)
        
        # State
        self.agent_pos = None
        self.remaining_goals = None
        self.step_count = 0
        
        # Render
        self.fig = None
        self.ax = None
        
        # Action mappings
        self._action_to_direction = {
            0: np.array([-1, 0]),  # Up
            1: np.array([1, 0]),   # Down
            2: np.array([0, -1]),  # Left
            3: np.array([0, 1])    # Right
        }
        
        # For stochastic dynamics
        self.rng = np.random.default_rng()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (can contain 'config' to override environment config)
        
        Returns:
            observation: Agent's position
            info: Additional information
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Allow dynamic configuration via options
        if options and 'config' in options:
            self.config = options['config']
            self.config.validate()
        
        # Reset state
        self.agent_pos = np.array(self.config.start_pos, dtype=np.int32)
        self.remaining_goals = set(self.config.goals)
        self.step_count = 0
        
        observation = self.agent_pos.copy()
        info = {
            'remaining_goals': len(self.remaining_goals),
            'total_goals': len(self.config.goals)
        }
        
        if self.render_mode == "human":
            self.render()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action to take (0-3)
        
        Returns:
            observation: New agent position
            reward: Reward received
            terminated: Whether episode ended (all goals reached)
            truncated: Whether episode was cut off (max steps reached)
            info: Additional information
        """
        self.step_count += 1
        
        # Apply stochastic dynamics
        actual_action = self._apply_dynamics(action)
        
        # Calculate new position
        direction = self._action_to_direction[actual_action]
        new_pos = self.agent_pos + direction
        
        # Check boundaries
        new_pos = np.clip(new_pos, 0, self.config.grid_size - 1)
        
        # Check for obstacles
        reward = 0.0
        if tuple(new_pos) in self.config.obstacles:
            # Hit obstacle - don't move, apply penalty
            reward += self.config.obstacle_penalty
        else:
            # Move to new position
            self.agent_pos = new_pos
            reward += self.config.step_penalty
        
        # Check if reached a goal
        current_pos_tuple = tuple(self.agent_pos)
        if current_pos_tuple in self.remaining_goals:
            reward += self.config.goal_reward
            self.remaining_goals.remove(current_pos_tuple)
        
        # Check termination conditions
        terminated = len(self.remaining_goals) == 0
        truncated = self.step_count >= self.config.max_steps
        
        observation = self.agent_pos.copy()
        info = {
            'remaining_goals': len(self.remaining_goals),
            'total_goals': len(self.config.goals),
            'step_count': self.step_count,
            'hit_obstacle': tuple(new_pos) in self.config.obstacles and not np.array_equal(new_pos, self.agent_pos)
        }
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_dynamics(self, action: int) -> int:
        """
        Apply stochastic dynamics to the action.
        
        Args:
            action: Intended action
        
        Returns:
            Actual action after applying dynamics
        """
        # Action success/failure
        if self.rng.random() > self.config.action_success_prob:
            # Action fails - stay in place (no movement)
            return action  # We'll handle this by not moving
        
        # Slip to adjacent direction
        if self.rng.random() < self.config.slip_prob:
            # Slip perpendicular to intended direction
            if action in [0, 1]:  # Up/Down - slip Left/Right
                return self.rng.choice([2, 3])
            else:  # Left/Right - slip Up/Down
                return self.rng.choice([0, 1])
        
        return action
    
    def render(self):
        """Visualize the environment."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            plt.ion()
        
        self.ax.clear()
        
        # Draw grid
        for i in range(self.config.grid_size + 1):
            self.ax.plot([0, self.config.grid_size], [i, i], 'k-', linewidth=0.5, alpha=0.3)
            self.ax.plot([i, i], [0, self.config.grid_size], 'k-', linewidth=0.5, alpha=0.3)
        
        # Draw obstacles (black squares)
        for obs in self.config.obstacles:
            obs_rect = patches.Rectangle(
                (obs[1], self.config.grid_size - 1 - obs[0]),
                1, 1,
                linewidth=1,
                edgecolor='black',
                facecolor='black',
                alpha=0.8
            )
            self.ax.add_patch(obs_rect)
        
        # Draw goals (red squares, dimmed if already reached)
        for goal in self.config.goals:
            alpha = 0.7 if goal in self.remaining_goals else 0.2
            goal_rect = patches.Rectangle(
                (goal[1], self.config.grid_size - 1 - goal[0]),
                1, 1,
                linewidth=2,
                edgecolor='red',
                facecolor='lightcoral',
                alpha=alpha
            )
            self.ax.add_patch(goal_rect)
        
        # Draw agent (green circle)
        agent_circle = patches.Circle(
            (self.agent_pos[1] + 0.5, self.config.grid_size - 1 - self.agent_pos[0] + 0.5),
            0.35,
            color='green',
            alpha=0.9,
            zorder=10
        )
        self.ax.add_patch(agent_circle)
        
        # Configuration
        self.ax.set_xlim(0, self.config.grid_size)
        self.ax.set_ylim(0, self.config.grid_size)
        self.ax.set_aspect('equal')
        
        title = f'Grid World {self.config.grid_size}x{self.config.grid_size} - Step {self.step_count}'
        title += f'\nGoals: {len(self.remaining_goals)}/{len(self.config.goals)} remaining'
        self.ax.set_title(title)
        
        self.ax.set_xticks(range(self.config.grid_size + 1))
        self.ax.set_yticks(range(self.config.grid_size + 1))
        self.ax.grid(False)
        
        # Add legend
        legend_elements = [
            patches.Patch(facecolor='green', alpha=0.9, label='Agent'),
            patches.Patch(facecolor='lightcoral', alpha=0.7, label='Goal'),
            patches.Patch(facecolor='black', alpha=0.8, label='Obstacle')
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def close(self):
        """Close the visualization window."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def state_to_index(self, state: np.ndarray) -> int:
        """Convert position (x, y) to unique index."""
        return state[0] * self.config.grid_size + state[1]
    
    def index_to_state(self, index: int) -> np.ndarray:
        """Convert index to position (x, y)."""
        x = index // self.config.grid_size
        y = index % self.config.grid_size
        return np.array([x, y], dtype=np.int32)
    
    def get_num_states(self) -> int:
        """Return total number of states."""
        return self.config.grid_size * self.config.grid_size


# Example usage
if __name__ == "__main__":
    # Example 1: Simple configuration
    config1 = GridWorldConfig(
        grid_size=8,
        start_pos=(0, 0),
        goals=[(7, 7), (7, 0), (0, 7)],
        obstacles=[(3, 3), (3, 4), (4, 3), (4, 4), (2, 5), (5, 2)],
        goal_reward=10.0,
        step_penalty=-0.1,
        obstacle_penalty=-1.0,
        max_steps=100
    )
    
    env = ConfigurableGridWorld(config=config1, render_mode="human")
    
    # Run a test episode
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    print(f"Info: {info}")
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not (terminated or truncated):
        # Random action
        action = np.random.randint(0, 4)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            print(f"\n🎉 All goals reached!")
        if truncated:
            print(f"\n⏰ Max steps reached!")
    
    print(f"Total reward: {total_reward:.2f}")
    print(f"Steps taken: {info['step_count']}")
    
    env.close()
    
    # Example 2: Stochastic dynamics
    config2 = GridWorldConfig(
        grid_size=5,
        start_pos=(2, 2),
        goals=[(0, 0)],
        obstacles=[(1, 2), (2, 1)],
        action_success_prob=0.8,  # 80% chance action succeeds
        slip_prob=0.1  # 10% chance of slipping
    )
    
    print("\n" + "="*50)
    print("Example with stochastic dynamics")
    print("="*50)
    
    env2 = ConfigurableGridWorld(config=config2, render_mode="human")
    obs, info = env2.reset(seed=123)
    
    for _ in range(10):
        action = 0  # Always try to go up
        obs, reward, terminated, truncated, info = env2.step(action)
        if terminated or truncated:
            break
    
    env2.close()