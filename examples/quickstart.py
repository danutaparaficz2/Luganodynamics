#!/usr/bin/env python3
"""
Quick start example for the robotic cable pickup simulation.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simulation import SimulationEnvironment
from src.visualization import SimulationVisualizer


def main():
    """Run a quick demonstration of the simulation."""
    # Create simulation environment
    sim = SimulationEnvironment(
        conveyor_length=2.0,
        conveyor_width=0.5,
        conveyor_speed=0.1
    )
    
    print("Robotic Cable Pickup Simulation - Quick Start")
    print("=" * 50)
    
    # Spawn cables
    print("\nSpawning 3 random cables...")
    for _ in range(3):
        sim.spawn_random_cable()
    
    # Run simulation
    print("Running simulation...")
    for i in range(50):
        # Try to grasp if cable is in pickup zone
        cables_in_zone = sim.conveyor.get_cables_in_pickup_zone(
            sim.pickup_zone_center,
            sim.pickup_zone_radius
        )
        
        action = 'grasp' if cables_in_zone and not sim.robotic_hand.is_gripping else None
        sim.step(action)
    
    # Print results
    print("\nResults:")
    print(f"  Simulation time: {sim.time:.2f}s")
    print(f"  Grasp attempts: {sim.grasp_attempts}")
    print(f"  Successful grasps: {sim.successful_grasps}")
    print(f"  Success rate: {sim.successful_grasps / max(1, sim.grasp_attempts) * 100:.1f}%")
    
    # Visualize
    print("\nGenerating visualization...")
    visualizer = SimulationVisualizer(sim)
    visualizer.plot_2d_snapshot(save_path='quickstart_result.png')
    
    print("\nVisualization saved to: quickstart_result.png")
    print("Done!")


if __name__ == "__main__":
    main()
