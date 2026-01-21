"""Cable object representation with flexible properties."""

import numpy as np
from typing import Tuple, List


class Cable:
    """Represents a cable with connector on the conveyor belt.
    
    The cable is modeled as an elongated, partly flexible object that can be
    curved or twisted. The connector is a rigid part at one end.
    """
    
    def __init__(
        self,
        connector_length: float = 0.05,  # 5cm connector
        connector_width: float = 0.02,   # 2cm connector width
        connector_height: float = 0.015, # 1.5cm connector height
        cable_length: float = 0.3,       # 30cm cable
        cable_diameter: float = 0.005,   # 5mm cable diameter
        flexibility: float = 0.3         # Cable flexibility (0=rigid, 1=very flexible)
    ):
        """Initialize cable with physical properties.
        
        Args:
            connector_length: Length of the connector in meters
            connector_width: Width of the connector in meters
            connector_height: Height of the connector in meters
            cable_length: Length of the cable in meters
            cable_diameter: Diameter of the cable in meters
            flexibility: How flexible the cable is (0-1)
        """
        self.connector_length = connector_length
        self.connector_width = connector_width
        self.connector_height = connector_height
        self.cable_length = cable_length
        self.cable_diameter = cable_diameter
        self.flexibility = flexibility
        
        # Position and orientation
        self.position = np.array([0.0, 0.0, 0.0])  # [x, y, z] in meters
        self.orientation = np.array([0.0, 0.0, 0.0])  # [roll, pitch, yaw] in radians
        
        # Cable shape (represented as control points)
        self.cable_points = self._generate_cable_shape()
        
    def _generate_cable_shape(self) -> np.ndarray:
        """Generate cable shape as a series of 3D points.
        
        Returns:
            Array of shape (N, 3) representing cable curve
        """
        # Number of points to represent the cable curve
        n_points = 20
        
        # Generate base cable path (straight line initially)
        t = np.linspace(0, 1, n_points)
        cable_points = np.zeros((n_points, 3))
        cable_points[:, 0] = t * self.cable_length  # x-direction
        
        # Add random curvature based on flexibility
        if self.flexibility > 0:
            # Add smooth random curves in y and z
            curve_amplitude = self.flexibility * self.cable_length * 0.1
            cable_points[:, 1] = curve_amplitude * np.sin(t * 2 * np.pi * np.random.rand())
            cable_points[:, 2] = curve_amplitude * 0.3 * np.sin(t * 3 * np.pi * np.random.rand())
        
        return cable_points
    
    def update_position(self, position: np.ndarray, orientation: np.ndarray):
        """Update cable position and orientation.
        
        Args:
            position: New position [x, y, z]
            orientation: New orientation [roll, pitch, yaw]
        """
        self.position = np.array(position)
        self.orientation = np.array(orientation)
        
    def get_connector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the pose (position and orientation) of the connector.
        
        This is the target for the robotic gripper.
        
        Returns:
            Tuple of (position, orientation) for the connector
        """
        # Connector is at the base of the cable
        connector_pos = self.position.copy()
        connector_ori = self.orientation.copy()
        
        return connector_pos, connector_ori
    
    def get_grasp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate optimal grasp pose for picking up the cable.
        
        Returns:
            Tuple of (position, orientation) for the grasp
        """
        connector_pos, connector_ori = self.get_connector_pose()
        
        # Grasp point is at the center of the connector
        grasp_pos = connector_pos.copy()
        grasp_pos[2] += self.connector_height / 2  # Grasp at mid-height
        
        # Grasp orientation aligned with connector
        grasp_ori = connector_ori.copy()
        
        return grasp_pos, grasp_ori
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the 3D bounding box of the entire cable assembly.
        
        Returns:
            Tuple of (min_bounds, max_bounds) where each is [x, y, z]
        """
        # Include connector dimensions
        connector_bounds = np.array([
            self.connector_length,
            self.connector_width,
            self.connector_height
        ])
        
        # Get cable bounds from control points
        cable_world_points = self.cable_points + self.position
        cable_min = np.min(cable_world_points, axis=0)
        cable_max = np.max(cable_world_points, axis=0)
        
        # Combine bounds
        min_bounds = np.minimum(self.position, cable_min)
        max_bounds = np.maximum(
            self.position + connector_bounds,
            cable_max
        )
        
        return min_bounds, max_bounds
    
    def randomize_shape(self):
        """Randomize the cable shape to simulate different configurations."""
        self.cable_points = self._generate_cable_shape()
