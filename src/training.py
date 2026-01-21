"""
Training module for the robotic hand using reinforcement learning principles.
"""
import numpy as np
from typing import List, Tuple
from src.simulation import SimulationEnvironment
from src.robotic_hand import GraspPose


class SimpleTrainer:
    """
    Simple trainer for the robotic hand using experience replay.
    This implements a basic training loop suitable for demonstration.
    """
    
    def __init__(self, simulation: SimulationEnvironment):
        """
        Initialize the trainer.
        
        Args:
            simulation: The simulation environment to train in
        """
        self.simulation = simulation
        self.experience_buffer = []
        self.max_buffer_size = 1000
        
    def collect_experience(self, num_episodes: int = 10) -> List[dict]:
        """
        Collect experience by running multiple episodes.
        
        Args:
            num_episodes: Number of episodes to run
            
        Returns:
            List of episode statistics
        """
        episode_stats = []
        
        for episode in range(num_episodes):
            stats = self.simulation.run_episode(num_cables=5, auto_grasp=True)
            episode_stats.append(stats)
            
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Success Rate = {stats['success_rate']:.2%}, "
                  f"Successful Grasps = {stats['successful_grasps']}/{stats['grasp_attempts']}")
        
        return episode_stats
    
    def train(self, num_episodes: int = 20) -> List[dict]:
        """
        Train the robotic hand over multiple episodes.
        
        Args:
            num_episodes: Number of training episodes
            
        Returns:
            List of episode statistics for analysis
        """
        print("Starting training...")
        print("=" * 50)
        
        all_episode_stats = []
        
        for episode in range(num_episodes):
            # Run episode
            stats = self.simulation.run_episode(num_cables=5, auto_grasp=True)
            all_episode_stats.append(stats)
            
            # Store experience
            self._store_experience(stats)
            
            # Print progress
            print(f"Episode {episode + 1}/{num_episodes}:")
            print(f"  Success Rate: {stats['success_rate']:.2%}")
            print(f"  Successful Grasps: {stats['successful_grasps']}")
            print(f"  Failed Grasps: {stats['failed_grasps']}")
            print("-" * 50)
        
        print("\nTraining completed!")
        print("=" * 50)
        
        # Calculate overall statistics
        total_attempts = sum(s['grasp_attempts'] for s in all_episode_stats)
        total_success = sum(s['successful_grasps'] for s in all_episode_stats)
        overall_success_rate = total_success / total_attempts if total_attempts > 0 else 0
        
        print(f"Overall Statistics:")
        print(f"  Total Grasp Attempts: {total_attempts}")
        print(f"  Total Successful Grasps: {total_success}")
        print(f"  Overall Success Rate: {overall_success_rate:.2%}")
        
        return all_episode_stats
    
    def _store_experience(self, episode_stats: dict):
        """
        Store experience from an episode.
        
        Args:
            episode_stats: Statistics from the episode
        """
        # Store observations from the episode
        for obs in episode_stats['observations']:
            if obs['num_cables_in_pickup_zone'] > 0:
                self.experience_buffer.append(obs)
        
        # Limit buffer size
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer = self.experience_buffer[-self.max_buffer_size:]
    
    def evaluate(self, num_episodes: int = 5) -> dict:
        """
        Evaluate the current policy.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation statistics
        """
        print("Evaluating current policy...")
        
        episode_stats = []
        for episode in range(num_episodes):
            stats = self.simulation.run_episode(num_cables=5, auto_grasp=True)
            episode_stats.append(stats)
        
        # Calculate average statistics
        avg_success_rate = np.mean([s['success_rate'] for s in episode_stats])
        avg_attempts = np.mean([s['grasp_attempts'] for s in episode_stats])
        
        eval_stats = {
            'avg_success_rate': avg_success_rate,
            'avg_attempts': avg_attempts,
            'episode_stats': episode_stats
        }
        
        print(f"Evaluation Results:")
        print(f"  Average Success Rate: {avg_success_rate:.2%}")
        print(f"  Average Grasp Attempts: {avg_attempts:.1f}")
        
        return eval_stats


class GraspPolicyImprover:
    """
    Improves grasp policy based on collected data.
    This is a simple heuristic-based improver for demonstration.
    """
    
    def __init__(self):
        """Initialize the policy improver."""
        self.successful_grasps = []
        self.failed_grasps = []
    
    def record_grasp(self, grasp_pose: GraspPose, success: bool):
        """
        Record a grasp attempt.
        
        Args:
            grasp_pose: The grasp pose that was attempted
            success: Whether the grasp was successful
        """
        if success:
            self.successful_grasps.append(grasp_pose)
        else:
            self.failed_grasps.append(grasp_pose)
    
    def get_statistics(self) -> dict:
        """
        Get statistics about recorded grasps.
        
        Returns:
            Dictionary with grasp statistics
        """
        total = len(self.successful_grasps) + len(self.failed_grasps)
        success_rate = len(self.successful_grasps) / total if total > 0 else 0
        
        return {
            'total_grasps': total,
            'successful_grasps': len(self.successful_grasps),
            'failed_grasps': len(self.failed_grasps),
            'success_rate': success_rate
        }
    
    def analyze_successful_patterns(self) -> dict:
        """
        Analyze patterns in successful grasps.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.successful_grasps:
            return {'message': 'No successful grasps to analyze'}
        
        # Analyze grip widths
        grip_widths = [g.grip_width for g in self.successful_grasps]
        confidences = [g.confidence for g in self.successful_grasps]
        
        analysis = {
            'avg_grip_width': np.mean(grip_widths),
            'std_grip_width': np.std(grip_widths),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
        
        return analysis
