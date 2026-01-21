"""
Visualization module for the robotic cable pickup simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional
from src.simulation import SimulationEnvironment


class SimulationVisualizer:
    """
    Visualizer for the robotic cable pickup simulation.
    """
    
    def __init__(self, simulation: SimulationEnvironment):
        """
        Initialize the visualizer.
        
        Args:
            simulation: The simulation environment to visualize
        """
        self.simulation = simulation
        self.fig = None
        self.ax = None
        
    def setup_3d_plot(self):
        """Setup a 3D plot for visualization."""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Robotic Cable Pickup Simulation')
        
    def setup_2d_plot(self):
        """Setup a 2D top-down plot for visualization."""
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Robotic Cable Pickup Simulation (Top View)')
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
    def plot_3d_snapshot(self, save_path: Optional[str] = None):
        """
        Plot a 3D snapshot of the current simulation state.
        
        Args:
            save_path: Optional path to save the figure
        """
        self.setup_3d_plot()
        
        # Draw conveyor belt
        min_bounds, max_bounds = self.simulation.conveyor.get_bounds()
        self._draw_conveyor_3d(min_bounds, max_bounds)
        
        # Draw pickup zone
        self._draw_pickup_zone_3d()
        
        # Draw cables
        for cable in self.simulation.conveyor.cables:
            self.ax.plot(cable.points[:, 0], cable.points[:, 1], cable.points[:, 2],
                        'b-', linewidth=3, label='Cable')
        
        # Draw robotic hand
        hand_pos = self.simulation.robotic_hand.position
        self.ax.scatter([hand_pos[0]], [hand_pos[1]], [hand_pos[2]],
                       c='red', s=200, marker='v', label='Robotic Hand')
        
        # Set axis limits
        self.ax.set_xlim([min_bounds[0] - 0.2, max_bounds[0] + 0.2])
        self.ax.set_ylim([min_bounds[1] - 0.2, max_bounds[1] + 0.2])
        self.ax.set_zlim([0, 0.7])
        
        # Add legend (only unique labels)
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.tight_layout()
        return self.fig
    
    def plot_2d_snapshot(self, save_path: Optional[str] = None):
        """
        Plot a 2D top-down snapshot of the current simulation state.
        
        Args:
            save_path: Optional path to save the figure
        """
        self.setup_2d_plot()
        
        # Draw conveyor belt
        min_bounds, max_bounds = self.simulation.conveyor.get_bounds()
        belt_rect = Rectangle(
            (min_bounds[0], min_bounds[1]),
            max_bounds[0] - min_bounds[0],
            max_bounds[1] - min_bounds[1],
            facecolor='gray', alpha=0.3, edgecolor='black', linewidth=2,
            label='Conveyor Belt'
        )
        self.ax.add_patch(belt_rect)
        
        # Draw pickup zone
        pickup_circle = plt.Circle(
            (self.simulation.pickup_zone_center[0], 
             self.simulation.pickup_zone_center[1]),
            self.simulation.pickup_zone_radius,
            facecolor='yellow', alpha=0.3, edgecolor='orange', linewidth=2,
            label='Pickup Zone'
        )
        self.ax.add_patch(pickup_circle)
        
        # Draw cables
        for i, cable in enumerate(self.simulation.conveyor.cables):
            self.ax.plot(cable.points[:, 0], cable.points[:, 1],
                        'b-', linewidth=3, label='Cable' if i == 0 else '')
        
        # Draw robotic hand
        hand_pos = self.simulation.robotic_hand.position
        self.ax.scatter([hand_pos[0]], [hand_pos[1]],
                       c='red', s=300, marker='v', label='Robotic Hand',
                       edgecolors='darkred', linewidths=2, zorder=5)
        
        # Set axis limits
        self.ax.set_xlim([min_bounds[0] - 0.3, max_bounds[0] + 0.3])
        self.ax.set_ylim([min_bounds[1] - 0.3, max_bounds[1] + 0.3])
        
        # Add statistics text
        stats_text = (f"Time: {self.simulation.time:.1f}s\n"
                     f"Cables: {len(self.simulation.conveyor.cables)}\n"
                     f"Grasps: {self.simulation.successful_grasps}/"
                     f"{self.simulation.grasp_attempts}\n"
                     f"Success Rate: {self.simulation.successful_grasps / max(1, self.simulation.grasp_attempts) * 100:.1f}%")
        
        self.ax.text(0.02, 0.98, stats_text,
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10)
        
        # Legend
        self.ax.legend(loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.tight_layout()
        return self.fig
    
    def _draw_conveyor_3d(self, min_bounds, max_bounds):
        """Draw conveyor belt in 3D plot."""
        # Draw the conveyor surface
        xx, yy = np.meshgrid(
            [min_bounds[0], max_bounds[0]],
            [min_bounds[1], max_bounds[1]]
        )
        zz = np.zeros_like(xx)
        self.ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')
    
    def _draw_pickup_zone_3d(self):
        """Draw pickup zone in 3D plot."""
        theta = np.linspace(0, 2*np.pi, 50)
        r = self.simulation.pickup_zone_radius
        x = self.simulation.pickup_zone_center[0] + r * np.cos(theta)
        y = self.simulation.pickup_zone_center[1] + r * np.sin(theta)
        z = np.zeros_like(x) + 0.001
        self.ax.plot(x, y, z, 'orange', linewidth=2, linestyle='--')
    
    def plot_training_metrics(self, episode_stats_list: list, save_path: Optional[str] = None):
        """
        Plot training metrics over multiple episodes.
        
        Args:
            episode_stats_list: List of episode statistics dictionaries
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        episodes = list(range(1, len(episode_stats_list) + 1))
        success_rates = [stats['success_rate'] for stats in episode_stats_list]
        grasp_attempts = [stats['grasp_attempts'] for stats in episode_stats_list]
        successful_grasps = [stats['successful_grasps'] for stats in episode_stats_list]
        
        # Success rate over episodes
        axes[0, 0].plot(episodes, success_rates, 'b-o', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_title('Grasp Success Rate Over Episodes')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1.1])
        
        # Total grasp attempts
        axes[0, 1].bar(episodes, grasp_attempts, color='orange', alpha=0.7)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Grasp Attempts')
        axes[0, 1].set_title('Total Grasp Attempts Per Episode')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Successful vs failed grasps
        failed_grasps = [stats['failed_grasps'] for stats in episode_stats_list]
        width = 0.35
        x = np.arange(len(episodes))
        axes[1, 0].bar(x - width/2, successful_grasps, width, label='Successful', color='green', alpha=0.7)
        axes[1, 0].bar(x + width/2, failed_grasps, width, label='Failed', color='red', alpha=0.7)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Successful vs Failed Grasps')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(episodes)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative success
        cumulative_success = np.cumsum(successful_grasps)
        axes[1, 1].plot(episodes, cumulative_success, 'g-o', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Cumulative Successful Grasps')
        axes[1, 1].set_title('Learning Progress (Cumulative Success)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved training metrics to {save_path}")
        
        return fig
