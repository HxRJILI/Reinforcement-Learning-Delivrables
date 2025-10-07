import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import pandas as pd
from qlearning_agent import QLearningAgent, train_agent, evaluate_agent
from configurable_gridworld import ConfigurableGridWorld, GridWorldConfig
import seaborn as sns
from matplotlib.gridspec import GridSpec


class HyperparameterSensitivityAnalysis:
    """
    Analyze sensitivity of Q-Learning convergence to hyperparameters.
    """
    
    def __init__(self, env_config: GridWorldConfig):
        """
        Initialize analysis with a fixed environment configuration.
        
        Args:
            env_config: GridWorld configuration to use for all experiments
        """
        self.env_config = env_config
        self.results = []
        
    def analyze_parameter(
        self,
        param_name: str,
        param_values: List[float],
        base_params: Dict[str, Any],
        num_episodes: int = 500,
        num_runs: int = 5,
        convergence_threshold: float = 0.9,
        eval_interval: int = 50
    ) -> pd.DataFrame:
        """
        Analyze sensitivity to a single parameter.
        
        Args:
            param_name: Name of parameter to vary
            param_values: List of values to test
            base_params: Base hyperparameters (dict with keys: learning_rate, 
                        discount_factor, epsilon, epsilon_min, epsilon_decay)
            num_episodes: Number of training episodes per run
            num_runs: Number of runs per parameter value (for averaging)
            convergence_threshold: Threshold for defining convergence (fraction of max reward)
            eval_interval: Evaluate every N episodes
            
        Returns:
            DataFrame with results
        """
        results = []
        
        print(f"\n{'='*60}")
        print(f"Analyzing parameter: {param_name}")
        print(f"{'='*60}")
        
        for param_value in param_values:
            print(f"\nTesting {param_name} = {param_value}")
            
            run_convergence_episodes = []
            run_convergence_steps = []
            run_final_rewards = []
            run_final_steps = []
            run_max_eval_rewards = []
            
            for run in range(num_runs):
                # Create environment
                env = ConfigurableGridWorld(config=self.env_config, render_mode=None)
                
                # Create agent with current parameter value
                agent_params = base_params.copy()
                agent_params[param_name] = param_value
                
                agent = QLearningAgent(
                    num_actions=4,
                    learning_rate=agent_params.get('learning_rate', 0.1),
                    discount_factor=agent_params.get('discount_factor', 0.99),
                    epsilon=agent_params.get('epsilon', 1.0),
                    epsilon_min=agent_params.get('epsilon_min', 0.01),
                    epsilon_decay=agent_params.get('epsilon_decay', 0.995),
                    learning_rate_decay=agent_params.get('learning_rate_decay', 1.0)
                )
                
                # Train agent
                metrics = train_agent(
                    env=env,
                    agent=agent,
                    num_episodes=num_episodes,
                    verbose=False,
                    eval_interval=eval_interval,
                    eval_episodes=10
                )
                
                # Analyze convergence
                eval_rewards = np.array(metrics['eval_rewards'])
                max_reward = np.max(eval_rewards)
                convergence_reward = convergence_threshold * max_reward
                
                # Find convergence point
                converged_idx = np.where(eval_rewards >= convergence_reward)[0]
                if len(converged_idx) > 0:
                    convergence_episode = metrics['eval_episodes'][converged_idx[0]]
                    convergence_step = agent.total_steps
                else:
                    convergence_episode = num_episodes
                    convergence_step = agent.total_steps
                
                # Final performance
                final_reward, final_step = evaluate_agent(env, agent, num_episodes=20)
                
                run_convergence_episodes.append(convergence_episode)
                run_convergence_steps.append(convergence_step)
                run_final_rewards.append(final_reward)
                run_final_steps.append(final_step)
                run_max_eval_rewards.append(max_reward)
                
                env.close()
                
                print(f"  Run {run+1}/{num_runs}: Converged at episode {convergence_episode}, "
                      f"Final reward: {final_reward:.3f}")
            
            # Store results
            results.append({
                'parameter': param_name,
                'value': param_value,
                'mean_convergence_episodes': np.mean(run_convergence_episodes),
                'std_convergence_episodes': np.std(run_convergence_episodes),
                'mean_convergence_steps': np.mean(run_convergence_steps),
                'std_convergence_steps': np.std(run_convergence_steps),
                'mean_final_reward': np.mean(run_final_rewards),
                'std_final_reward': np.std(run_final_rewards),
                'mean_final_steps': np.mean(run_final_steps),
                'std_final_steps': np.std(run_final_steps),
                'mean_max_eval_reward': np.mean(run_max_eval_rewards),
                'std_max_eval_reward': np.std(run_max_eval_rewards)
            })
        
        df = pd.DataFrame(results)
        self.results.append(df)
        return df
    
    def run_full_analysis(
        self,
        param_ranges: Dict[str, List[float]],
        base_params: Dict[str, Any],
        num_episodes: int = 500,
        num_runs: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Run sensitivity analysis for multiple parameters.
        
        Args:
            param_ranges: Dict mapping parameter names to lists of values to test
            base_params: Base hyperparameters
            num_episodes: Number of training episodes per run
            num_runs: Number of runs per parameter value
            
        Returns:
            Dictionary mapping parameter names to result DataFrames
        """
        all_results = {}
        
        for param_name, param_values in param_ranges.items():
            df = self.analyze_parameter(
                param_name=param_name,
                param_values=param_values,
                base_params=base_params,
                num_episodes=num_episodes,
                num_runs=num_runs
            )
            all_results[param_name] = df
        
        return all_results
    
    def plot_sensitivity(self, results_df: pd.DataFrame, save_path: str = None):
        """
        Plot sensitivity analysis results for a single parameter.
        
        Args:
            results_df: DataFrame from analyze_parameter
            save_path: Optional path to save figure
        """
        param_name = results_df['parameter'].iloc[0]
        
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Convergence episodes vs parameter
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.errorbar(
            results_df['value'],
            results_df['mean_convergence_episodes'],
            yerr=results_df['std_convergence_episodes'],
            marker='o', capsize=5, capthick=2, linewidth=2, markersize=8
        )
        ax1.set_xlabel(param_name, fontsize=12)
        ax1.set_ylabel('Episodes to Convergence', fontsize=12)
        ax1.set_title(f'Convergence Speed vs {param_name}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Total steps to convergence vs parameter
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.errorbar(
            results_df['value'],
            results_df['mean_convergence_steps'],
            yerr=results_df['std_convergence_steps'],
            marker='s', capsize=5, capthick=2, linewidth=2, markersize=8, color='orange'
        )
        ax2.set_xlabel(param_name, fontsize=12)
        ax2.set_ylabel('Total Steps to Convergence', fontsize=12)
        ax2.set_title(f'Training Steps vs {param_name}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Final reward vs parameter
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.errorbar(
            results_df['value'],
            results_df['mean_final_reward'],
            yerr=results_df['std_final_reward'],
            marker='^', capsize=5, capthick=2, linewidth=2, markersize=8, color='green'
        )
        ax3.set_xlabel(param_name, fontsize=12)
        ax3.set_ylabel('Final Average Reward', fontsize=12)
        ax3.set_title(f'Final Performance vs {param_name}', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Final steps vs parameter
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.errorbar(
            results_df['value'],
            results_df['mean_final_steps'],
            yerr=results_df['std_final_steps'],
            marker='d', capsize=5, capthick=2, linewidth=2, markersize=8, color='red'
        )
        ax4.set_xlabel(param_name, fontsize=12)
        ax4.set_ylabel('Steps per Episode (Final)', fontsize=12)
        ax4.set_title(f'Efficiency vs {param_name}', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Hyperparameter Sensitivity Analysis: {param_name}',
                     fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.close(fig)  # Close figure to free memory
    
    def plot_comparison(self, all_results: Dict[str, pd.DataFrame], save_path: str = None):
        """
        Plot comparison of all parameters on a single figure.
        
        Args:
            all_results: Dictionary mapping parameter names to result DataFrames
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hyperparameter Sensitivity Comparison', fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        
        for idx, (param_name, df) in enumerate(all_results.items()):
            color = colors[idx]
            
            # Normalize values for comparison
            values_norm = (df['value'] - df['value'].min()) / (df['value'].max() - df['value'].min())
            
            # Plot convergence episodes
            axes[0, 0].plot(values_norm, df['mean_convergence_episodes'], 
                           marker='o', label=param_name, color=color, linewidth=2)
            axes[0, 0].fill_between(
                values_norm,
                df['mean_convergence_episodes'] - df['std_convergence_episodes'],
                df['mean_convergence_episodes'] + df['std_convergence_episodes'],
                alpha=0.2, color=color
            )
            
            # Plot convergence steps
            axes[0, 1].plot(values_norm, df['mean_convergence_steps'],
                           marker='s', label=param_name, color=color, linewidth=2)
            
            # Plot final reward
            axes[1, 0].plot(values_norm, df['mean_final_reward'],
                           marker='^', label=param_name, color=color, linewidth=2)
            
            # Plot final steps
            axes[1, 1].plot(values_norm, df['mean_final_steps'],
                           marker='d', label=param_name, color=color, linewidth=2)
        
        axes[0, 0].set_ylabel('Episodes to Convergence', fontsize=11)
        axes[0, 0].set_xlabel('Normalized Parameter Value', fontsize=11)
        axes[0, 0].set_title('Convergence Speed', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_ylabel('Total Steps to Convergence', fontsize=11)
        axes[0, 1].set_xlabel('Normalized Parameter Value', fontsize=11)
        axes[0, 1].set_title('Training Steps', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_ylabel('Final Average Reward', fontsize=11)
        axes[1, 0].set_xlabel('Normalized Parameter Value', fontsize=11)
        axes[1, 0].set_title('Final Performance', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_ylabel('Steps per Episode', fontsize=11)
        axes[1, 1].set_xlabel('Normalized Parameter Value', fontsize=11)
        axes[1, 1].set_title('Efficiency', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison figure saved to {save_path}")
        
        plt.close(fig)  # Close figure to free memory
    
    def generate_report(self, all_results: Dict[str, pd.DataFrame], save_path: str = None):
        """
        Generate a text report summarizing the sensitivity analysis.
        
        Args:
            all_results: Dictionary mapping parameter names to result DataFrames
            save_path: Optional path to save report
        """
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("HYPERPARAMETER SENSITIVITY ANALYSIS REPORT")
        report_lines.append("="*70)
        report_lines.append("")
        
        # Environment info
        report_lines.append("Environment Configuration:")
        report_lines.append(f"  Grid Size: {self.env_config.grid_size}x{self.env_config.grid_size}")
        report_lines.append(f"  Start Position: {self.env_config.start_pos}")
        report_lines.append(f"  Goals: {self.env_config.goals}")
        report_lines.append(f"  Obstacles: {self.env_config.obstacles}")
        report_lines.append("")
        
        for param_name, df in all_results.items():
            report_lines.append("-"*70)
            report_lines.append(f"Parameter: {param_name}")
            report_lines.append("-"*70)
            
            # Find best value for each metric
            best_convergence = df.loc[df['mean_convergence_episodes'].idxmin()]
            best_reward = df.loc[df['mean_final_reward'].idxmax()]
            best_steps = df.loc[df['mean_final_steps'].idxmin()]
            
            report_lines.append(f"\nBest for Convergence Speed:")
            report_lines.append(f"  Value: {best_convergence['value']}")
            report_lines.append(f"  Episodes: {best_convergence['mean_convergence_episodes']:.1f} ± "
                              f"{best_convergence['std_convergence_episodes']:.1f}")
            
            report_lines.append(f"\nBest for Final Reward:")
            report_lines.append(f"  Value: {best_reward['value']}")
            report_lines.append(f"  Reward: {best_reward['mean_final_reward']:.3f} ± "
                              f"{best_reward['std_final_reward']:.3f}")
            
            report_lines.append(f"\nBest for Efficiency (fewer steps):")
            report_lines.append(f"  Value: {best_steps['value']}")
            report_lines.append(f"  Steps: {best_steps['mean_final_steps']:.1f} ± "
                              f"{best_steps['std_final_steps']:.1f}")
            
            report_lines.append(f"\nAll tested values:")
            for _, row in df.iterrows():
                report_lines.append(f"  {param_name}={row['value']}: "
                                  f"Conv={row['mean_convergence_episodes']:.1f}, "
                                  f"Reward={row['mean_final_reward']:.3f}, "
                                  f"Steps={row['mean_final_steps']:.1f}")
            report_lines.append("")
        
        report_lines.append("="*70)
        
        report_text = "\n".join(report_lines)
        print(report_text)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to {save_path}")


# Example usage
if __name__ == "__main__":
    # Define fixed environment
    config = GridWorldConfig(
        grid_size=15,
        start_pos=(0, 0),
        goals=[(10, 10)],
        obstacles=[(2, 2), (2, 3), (3, 2)],
        goal_reward=1.0,
        step_penalty=-0.01,
        obstacle_penalty=-0.5,
        max_steps=100
    )
    
    # Create analyzer
    analyzer = HyperparameterSensitivityAnalysis(env_config=config)
    
    # Define base parameters
    base_params = {
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'learning_rate_decay': 1.0
    }
    
    # Define parameter ranges to test
    param_ranges = {
        'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5],
        'discount_factor': [0.8, 0.9, 0.95, 0.99, 0.999],
        'epsilon_decay': [0.95, 0.97, 0.99, 0.995, 0.999]
    }
    
    # Run full analysis
    print("Starting hyperparameter sensitivity analysis...")
    print("This may take several minutes...")
    
    all_results = analyzer.run_full_analysis(
        param_ranges=param_ranges,
        base_params=base_params,
        num_episodes=500,
        num_runs=3  # Increase for more robust results
    )
    
    # Generate visualizations
    print("\n\nGenerating visualizations...")
    
    # Individual plots for each parameter
    for param_name, df in all_results.items():
        analyzer.plot_sensitivity(df, save_path=f"sensitivity_{param_name}.png")
    
    # Comparison plot
    analyzer.plot_comparison(all_results, save_path="sensitivity_comparison.png")
    
    # Generate report
    analyzer.generate_report(all_results, save_path="sensitivity_report.txt")
    
    print("\n\nAnalysis complete!")