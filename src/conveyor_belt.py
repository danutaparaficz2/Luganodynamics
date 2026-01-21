"""
Conveyor belt module for the robotic cable pickup simulation.
"""
import numpy as np
from typing import List, Tuple
from src.cable import Cable


class ConveyorBelt:
    """
    Represents a conveyor belt carrying cables.
    """
    
    def __init__(self, length: float = 2.0, width: float = 0.5, 
                 speed: float = 0.1, position: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        Initialize a conveyor belt.
        
        Args:
            length: Length of the conveyor belt in meters
            width: Width of the conveyor belt in meters
            speed: Speed of the belt in meters per second
            position: Position (x, y, z) of the belt center
        """
        self.length = length
        self.width = width
        self.speed = speed
        self.position = np.array(position, dtype=np.float32)
        self.cables: List[Cable] = []
        self.time = 0.0
        
    def add_cable(self, cable: Cable):
        """Add a cable to the conveyor belt."""
        self.cables.append(cable)
        
    def spawn_cable(self, cable_length: float = 1.0) -> Cable:
        """
        Spawn a new cable at the start of the conveyor belt.
        
        Args:
            cable_length: Length of the cable to spawn
            
        Returns:
            The newly created cable
        """
        # Spawn at the beginning of the belt
        spawn_position = self.position + np.array([-self.length/2, 0.0, 0.0])
        cable = Cable(length=cable_length, position=tuple(spawn_position))
        self.add_cable(cable)
        return cable
    
    def update(self, dt: float):
        """
        Update the conveyor belt state.
        
        Args:
            dt: Time step in seconds
        """
        self.time += dt
        
        # Move all cables along the belt
        movement = np.array([self.speed * dt, 0.0, 0.0])
        for cable in self.cables:
            cable.position += movement
            cable.points += movement
            
        # Remove cables that have passed the end of the belt
        end_position = self.position[0] + self.length / 2
        self.cables = [cable for cable in self.cables 
                       if cable.position[0] < end_position + 0.5]
    
    def get_cables_in_pickup_zone(self, pickup_zone_center: np.ndarray, 
                                   pickup_zone_radius: float = 0.3) -> List[Cable]:
        """
        Get all cables currently in the pickup zone.
        
        Args:
            pickup_zone_center: Center position of the pickup zone
            pickup_zone_radius: Radius of the pickup zone
            
        Returns:
            List of cables in the pickup zone
        """
        cables_in_zone = []
        for cable in self.cables:
            distance = np.linalg.norm(cable.get_centerpoint()[:2] - pickup_zone_center[:2])
            if distance < pickup_zone_radius:
                cables_in_zone.append(cable)
        return cables_in_zone
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the bounds of the conveyor belt."""
        half_length = self.length / 2
        half_width = self.width / 2
        
        min_bounds = self.position + np.array([-half_length, -half_width, 0.0])
        max_bounds = self.position + np.array([half_length, half_width, 0.1])
        
        return min_bounds, max_bounds
