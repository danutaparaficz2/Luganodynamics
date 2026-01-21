"""
Main simulation environment for robotic cable pickup.
"""
import numpy as np
from typing import List, Optional, Tuple
from src.cable import Cable
from src.conveyor_belt import ConveyorBelt
from src.robotic_hand import RoboticHand, GraspPose


class SimulationEnvironment:
    """
    Main simulation environment that integrates cable, conveyor belt, and robotic hand.
    """
    
    def __init__(self, conveyor_length: float = 2.0, conveyor_width: float = 0.5,
                 conveyor_speed: float = 0.1):
        """
        Initialize the simulation environment.
        
        Args:
            conveyor_length: Length of conveyor belt in meters
            conveyor_width: Width of conveyor belt in meters
            conveyor_speed: Speed of conveyor belt in m/s
        """
        self.conveyor = ConveyorBelt(
            length=conveyor_length,
            width=conveyor_width,
            speed=conveyor_speed
        )
        
        # Position robotic hand above the middle of the conveyor
        hand_position = (0.0, 0.0, 0.5)
        self.robotic_hand = RoboticHand(position=hand_position)
        
        self.pickup_zone_center = np.array([0.0, 0.0, 0.0])
        self.pickup_zone_radius = 0.3
        
        self.time = 0.0
        self.dt = 0.05  # 50ms time step
        
        # Training metrics
        self.grasp_attempts = 0
        self.successful_grasps = 0
        self.failed_grasps = 0
        
    def reset(self):
        """Reset the simulation to initial state."""
        self.conveyor.cables.clear()
        self.robotic_hand.release()
        self.time = 0.0
        self.grasp_attempts = 0
        self.successful_grasps = 0
        self.failed_grasps = 0
        
    def spawn_random_cable(self, cable_length: float = None) -> Cable:
        """
        Spawn a cable with random shape on the conveyor.
        
        Args:
            cable_length: Length of cable, or None for random length
            
        Returns:
            The spawned cable
        """
        if cable_length is None:
            cable_length = np.random.uniform(0.5, 1.5)
        
        cable = self.conveyor.spawn_cable(cable_length=cable_length)
        # Generate random shape
        cable.generate_shape()
        return cable
    
    def step(self, action: Optional[str] = None) -> dict:
        """
        Advance the simulation by one time step.
        
        Args:
            action: Action to take ('grasp', 'release', or None)
            
        Returns:
            Dictionary containing observation data
        """
        # Update conveyor belt
        self.conveyor.update(self.dt)
        self.time += self.dt
        
        # Execute action if provided
        if action == 'grasp':
            self._attempt_grasp()
        elif action == 'release':
            self.robotic_hand.release()
        
        # Get observation
        observation = self._get_observation()
        
        return observation
    
    def _attempt_grasp(self) -> bool:
        """
        Attempt to grasp a cable in the pickup zone.
        
        Returns:
            True if grasp was successful, False otherwise
        """
        # Find cables in pickup zone
        cables_in_zone = self.conveyor.get_cables_in_pickup_zone(
            self.pickup_zone_center,
            self.pickup_zone_radius
        )
        
        if not cables_in_zone:
            return False
        
        # Select the cable closest to the pickup zone center
        target_cable = min(cables_in_zone, 
                          key=lambda c: np.linalg.norm(
                              c.get_centerpoint() - self.pickup_zone_center))
        
        # Compute grasp pose
        grasp_pose = self.robotic_hand.compute_grasp_pose(target_cable)
        
        if grasp_pose is None:
            return False
        
        # Execute grasp
        self.grasp_attempts += 1
        success = self.robotic_hand.execute_grasp(grasp_pose, target_cable)
        
        if success:
            self.successful_grasps += 1
            # Remove cable from conveyor (picked up)
            self.conveyor.cables.remove(target_cable)
        else:
            self.failed_grasps += 1
        
        return success
    
    def _get_observation(self) -> dict:
        """
        Get current observation of the environment.
        
        Returns:
            Dictionary with observation data
        """
        cables_in_zone = self.conveyor.get_cables_in_pickup_zone(
            self.pickup_zone_center,
            self.pickup_zone_radius
        )
        
        observation = {
            'time': self.time,
            'num_cables_on_belt': len(self.conveyor.cables),
            'num_cables_in_pickup_zone': len(cables_in_zone),
            'robotic_hand_state': self.robotic_hand.get_state(),
            'cables_in_zone': cables_in_zone,
            'grasp_success_rate': (self.successful_grasps / self.grasp_attempts 
                                  if self.grasp_attempts > 0 else 0.0)
        }
        
        return observation
    
    def run_episode(self, num_cables: int = 5, auto_grasp: bool = True) -> dict:
        """
        Run a full episode of the simulation.
        
        Args:
            num_cables: Number of cables to spawn during episode
            auto_grasp: Whether to automatically attempt grasps
            
        Returns:
            Dictionary with episode statistics
        """
        self.reset()
        
        cable_spawn_interval = 2.0  # Spawn cable every 2 seconds
        next_spawn_time = 0.0
        cables_spawned = 0
        
        observations = []
        
        # Run until all cables have been processed
        max_time = num_cables * cable_spawn_interval + 10.0
        
        while self.time < max_time:
            # Spawn cables at intervals
            if cables_spawned < num_cables and self.time >= next_spawn_time:
                self.spawn_random_cable()
                cables_spawned += 1
                next_spawn_time += cable_spawn_interval
            
            # Auto grasp if enabled
            action = None
            if auto_grasp:
                cables_in_zone = self.conveyor.get_cables_in_pickup_zone(
                    self.pickup_zone_center,
                    self.pickup_zone_radius
                )
                if cables_in_zone and not self.robotic_hand.is_gripping:
                    action = 'grasp'
            
            # Step simulation
            obs = self.step(action)
            observations.append(obs)
        
        # Compile episode statistics
        episode_stats = {
            'total_time': self.time,
            'cables_spawned': cables_spawned,
            'grasp_attempts': self.grasp_attempts,
            'successful_grasps': self.successful_grasps,
            'failed_grasps': self.failed_grasps,
            'success_rate': (self.successful_grasps / self.grasp_attempts 
                           if self.grasp_attempts > 0 else 0.0),
            'observations': observations
        }
        
        return episode_stats
    
    def get_state_for_training(self) -> np.ndarray:
        """
        Get state representation suitable for training a neural network.
        
        Returns:
            State vector as numpy array
        """
        # Get cables in pickup zone
        cables_in_zone = self.conveyor.get_cables_in_pickup_zone(
            self.pickup_zone_center,
            self.pickup_zone_radius
        )
        
        # Default state if no cables
        if not cables_in_zone:
            return np.zeros(10, dtype=np.float32)
        
        # Use the closest cable
        target_cable = min(cables_in_zone,
                          key=lambda c: np.linalg.norm(
                              c.get_centerpoint() - self.pickup_zone_center))
        
        # Extract features
        cable_center = target_cable.get_centerpoint()
        cable_orientation = target_cable.get_orientation()
        hand_position = self.robotic_hand.position
        
        # Create state vector
        state = np.concatenate([
            cable_center,           # 3 values
            cable_orientation,      # 3 values
            hand_position,          # 3 values
            [self.robotic_hand.current_grip_width]  # 1 value
        ])
        
        return state.astype(np.float32)
