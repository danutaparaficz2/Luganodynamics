#!/usr/bin/env python3
"""Example script to run the robotic cable pickup simulation."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simulation.environment import SimulationEnvironment
from simulation.cable import Cable
from utils.config import (
    NUM_TRAINING_EPISODES,
    CABLES_PER_EPISODE,
    MAX_STEPS_PER_EPISODE
)


def main():
    """Run the simulation."""
    print("=" * 60)
    print("Robotic Hand Cable Pickup Simulation")
    print("Simulation-First Training Approach")
    print("=" * 60)
    print()
    
    # Create simulation environment
    print("Initializing simulation environment...")
    env = SimulationEnvironment(use_gui=False)
    print("✓ Environment initialized")
    print()
    
    # Run single episode for demonstration
    print(f"Running demonstration episode with {CABLES_PER_EPISODE} cables...")
    print()
    
    results = env.run_episode(
        num_cables=CABLES_PER_EPISODE,
        max_steps=MAX_STEPS_PER_EPISODE
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("Episode Results")
    print("=" * 60)
    print(f"Episode:             {results['episode']}")
    print(f"Simulation time:     {results['time']:.2f} seconds")
    print(f"Total pickups:       {results['total_pickups']}")
    print(f"Successful pickups:  {results['successful_pickups']}")
    print(f"Failed pickups:      {results['failed_pickups']}")
    print(f"Success rate:        {results['success_rate']:.1%}")
    print("=" * 60)
    print()
    
    # Run training episodes
    print(f"Running {NUM_TRAINING_EPISODES} training episodes...")
    print()
    
    all_results = []
    for episode_num in range(NUM_TRAINING_EPISODES):
        results = env.run_episode(
            num_cables=CABLES_PER_EPISODE,
            max_steps=MAX_STEPS_PER_EPISODE
        )
        all_results.append(results)
        
        # Print progress every 10 episodes
        if (episode_num + 1) % 10 == 0:
            avg_success = sum(r['success_rate'] for r in all_results[-10:]) / 10
            print(f"Episodes {episode_num-8:3d}-{episode_num+1:3d}: "
                  f"Avg success rate = {avg_success:.1%}")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    
    total_attempts = sum(r['total_pickups'] for r in all_results)
    total_success = sum(r['successful_pickups'] for r in all_results)
    overall_success_rate = total_success / total_attempts if total_attempts > 0 else 0
    
    print(f"Total episodes:      {NUM_TRAINING_EPISODES}")
    print(f"Total attempts:      {total_attempts}")
    print(f"Total successes:     {total_success}")
    print(f"Overall success:     {overall_success_rate:.1%}")
    print("=" * 60)
    print()
    
    print("Simulation completed successfully!")
    print()
    print("Next steps:")
    print("1. Use collected data to train a neural network policy")
    print("2. Implement more advanced vision algorithms for pose estimation")
    print("3. Add noise and variations to test robustness")
    print("4. Integrate with real robot hardware")
    

if __name__ == "__main__":
    main()
