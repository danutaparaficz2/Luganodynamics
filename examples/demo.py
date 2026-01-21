#!/usr/bin/env python3
"""
Demo script for the robotic cable pickup simulation.
Run this script to see the simulation in action with visualization.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simulation import SimulationEnvironment
from src.visualization import SimulationVisualizer
from src.training import SimpleTrainer
import matplotlib.pyplot as plt


def demo_basic_simulation():
    """Demonstrate basic simulation with visualization."""
    print("=" * 60)
    print("Robotic Cable Pickup Simulation - Basic Demo")
    print("=" * 60)
    
    # Create simulation
    sim = SimulationEnvironment(
        conveyor_length=2.0,
        conveyor_width=0.5,
        conveyor_speed=0.1
    )
    
    # Spawn some cables
    print("\nSpawning cables on conveyor belt...")
    for i in range(3):
        sim.spawn_random_cable()
        print(f"  Cable {i+1} spawned")
    
    # Create visualizer
    visualizer = SimulationVisualizer(sim)
    
    # Show initial state
    print("\nGenerating visualization of initial state...")
    visualizer.plot_2d_snapshot(save_path='simulation_initial.png')
    visualizer.plot_3d_snapshot(save_path='simulation_3d.png')
    
    print("\nRunning simulation steps...")
    # Run some steps with automatic grasping
    for step in range(100):
        # Check for cables in pickup zone
        cables_in_zone = sim.conveyor.get_cables_in_pickup_zone(
            sim.pickup_zone_center,
            sim.pickup_zone_radius
        )
        
        action = None
        if cables_in_zone and not sim.robotic_hand.is_gripping:
            action = 'grasp'
            print(f"  Step {step}: Attempting grasp...")
        
        obs = sim.step(action)
        
        # Visualize after first grasp
        if step == 30:
            visualizer.plot_2d_snapshot(save_path='simulation_grasp.png')
    
    print("\nSimulation complete!")
    print(f"Total time simulated: {sim.time:.2f} seconds")
    print(f"Grasp attempts: {sim.grasp_attempts}")
    print(f"Successful grasps: {sim.successful_grasps}")
    print(f"Success rate: {sim.successful_grasps / max(1, sim.grasp_attempts) * 100:.1f}%")
    
    # Show final state
    visualizer.plot_2d_snapshot(save_path='simulation_final.png')
    
    print("\nVisualization saved to:")
    print("  - simulation_initial.png (2D top view - initial)")
    print("  - simulation_3d.png (3D view)")
    print("  - simulation_grasp.png (2D top view - during grasp)")
    print("  - simulation_final.png (2D top view - final)")


def demo_training():
    """Demonstrate training the robotic hand."""
    print("\n" + "=" * 60)
    print("Robotic Cable Pickup Simulation - Training Demo")
    print("=" * 60)
    
    # Create simulation
    sim = SimulationEnvironment()
    
    # Create trainer
    trainer = SimpleTrainer(sim)
    
    # Run training
    print("\nTraining robotic hand over 10 episodes...")
    episode_stats = trainer.train(num_episodes=10)
    
    # Visualize training progress
    visualizer = SimulationVisualizer(sim)
    print("\nGenerating training metrics visualization...")
    visualizer.plot_training_metrics(episode_stats, save_path='training_metrics.png')
    
    print("\nTraining metrics saved to: training_metrics.png")
    
    # Evaluate
    print("\n" + "-" * 60)
    eval_stats = trainer.evaluate(num_episodes=5)
    
    return episode_stats


def demo_single_episode():
    """Run a single episode with detailed output."""
    print("\n" + "=" * 60)
    print("Robotic Cable Pickup Simulation - Single Episode Demo")
    print("=" * 60)
    
    # Create simulation
    sim = SimulationEnvironment(
        conveyor_length=2.0,
        conveyor_width=0.5,
        conveyor_speed=0.15
    )
    
    # Run episode
    print("\nRunning single episode with 5 cables...")
    stats = sim.run_episode(num_cables=5, auto_grasp=True)
    
    print("\nEpisode Results:")
    print(f"  Total Time: {stats['total_time']:.2f} seconds")
    print(f"  Cables Spawned: {stats['cables_spawned']}")
    print(f"  Grasp Attempts: {stats['grasp_attempts']}")
    print(f"  Successful Grasps: {stats['successful_grasps']}")
    print(f"  Failed Grasps: {stats['failed_grasps']}")
    print(f"  Success Rate: {stats['success_rate']:.2%}")
    
    # Visualize final state
    visualizer = SimulationVisualizer(sim)
    visualizer.plot_2d_snapshot(save_path='episode_result.png')
    print("\nEpisode visualization saved to: episode_result.png")
    
    return stats


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ROBOTIC CABLE PICKUP SIMULATION")
    print("Simulation-First Approach for Training Robotic Hand")
    print("=" * 60)
    
    # Run demos
    demo_basic_simulation()
    demo_single_episode()
    demo_training()
    
    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("Check the generated PNG files for visualizations.")
    print("=" * 60)
