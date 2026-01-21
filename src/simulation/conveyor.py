"""Conveyor belt simulation with moving cables."""

import numpy as np
from typing import List, Optional

try:
    from .cable import Cable
except ImportError:
    from simulation.cable import Cable


class ConveyorBelt:
    """Simulates a conveyor belt that moves cables.
    
    The belt moves at a constant speed, and cables are placed on it
    at random intervals.
    """
    
    def __init__(
        self,
        length: float = 2.0,       # 2m belt length
        width: float = 0.5,        # 0.5m belt width
        height: float = 0.8,       # 0.8m belt height from ground
        speed: float = 0.05,       # 5 cm/s belt speed
        pickup_position: float = 1.0  # Position along belt where robot picks up
    ):
        """Initialize conveyor belt.
        
        Args:
            length: Length of the belt in meters
            width: Width of the belt in meters
            height: Height of the belt from ground in meters
            speed: Speed of the belt in m/s
            pickup_position: X position along belt where robot should pick up
        """
        self.length = length
        self.width = width
        self.height = height
        self.speed = speed
        self.pickup_position = pickup_position
        
        # Cables currently on the belt
        self.cables: List[Cable] = []
        
        # Belt position (for animation)
        self.belt_offset = 0.0
        
        # Simulation time
        self.time = 0.0
        
    def add_cable(self, cable: Cable, x_position: Optional[float] = None):
        """Add a cable to the belt.
        
        Args:
            cable: Cable object to add
            x_position: Initial x position on belt (None for start of belt)
        """
        if x_position is None:
            x_position = 0.0
            
        # Place cable on belt surface
        y_position = 0.0  # Center of belt
        z_position = self.height + cable.connector_height / 2
        
        # Random orientation (mostly aligned with belt, slight variations)
        orientation = np.array([
            np.random.uniform(-0.1, 0.1),  # Small roll
            np.random.uniform(-0.1, 0.1),  # Small pitch
            np.random.uniform(-0.2, 0.2)   # Some yaw variation
        ])
        
        cable.update_position(
            np.array([x_position, y_position, z_position]),
            orientation
        )
        
        # Randomize cable shape
        cable.randomize_shape()
        
        self.cables.append(cable)
        
    def update(self, dt: float):
        """Update belt state by one time step.
        
        Args:
            dt: Time step in seconds
        """
        self.time += dt
        self.belt_offset += self.speed * dt
        
        # Move all cables with the belt
        for cable in self.cables:
            cable.position[0] += self.speed * dt
            
        # Remove cables that have moved off the belt
        self.cables = [
            cable for cable in self.cables
            if cable.position[0] < self.length
        ]
        
    def get_cable_at_pickup(self) -> Optional[Cable]:
        """Get cable currently at pickup position, if any.
        
        Returns:
            Cable at pickup position, or None
        """
        # Tolerance for pickup position
        tolerance = 0.1  # 10cm
        
        for cable in self.cables:
            if abs(cable.position[0] - self.pickup_position) < tolerance:
                return cable
                
        return None
    
    def remove_cable(self, cable: Cable):
        """Remove a cable from the belt (after pickup).
        
        Args:
            cable: Cable to remove
        """
        if cable in self.cables:
            self.cables.remove(cable)
            
    def get_all_cables(self) -> List[Cable]:
        """Get all cables currently on the belt.
        
        Returns:
            List of cables on the belt
        """
        return self.cables.copy()
