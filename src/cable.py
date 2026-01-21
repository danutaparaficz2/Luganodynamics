"""
Cable module for representing flexible elongated objects on conveyor belt.
"""
import numpy as np
from typing import List, Tuple


class Cable:
    """
    Represents a flexible cable with variable shape on the conveyor belt.
    The cable is modeled as a series of connected points forming a curve.
    """
    
    def __init__(self, length: float = 1.0, num_segments: int = 20, 
                 position: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        Initialize a cable object.
        
        Args:
            length: Total length of the cable in meters
            num_segments: Number of segments to discretize the cable
            position: Initial position (x, y, z) of the cable center
        """
        self.length = length
        self.num_segments = num_segments
        self.position = np.array(position, dtype=np.float32)
        self.points = None
        self.generate_shape()
        
    def generate_shape(self, curvature: float = None, twist: float = None):
        """
        Generate the cable shape with optional curvature and twist.
        
        Args:
            curvature: Amount of curve (0 = straight, higher = more curved)
            twist: Amount of twist along the cable axis
        """
        if curvature is None:
            curvature = np.random.uniform(0.0, 2.0)
        if twist is None:
            twist = np.random.uniform(-np.pi, np.pi)
            
        # Generate parametric curve for cable shape
        t = np.linspace(0, 1, self.num_segments)
        segment_length = self.length / (self.num_segments - 1)
        
        # Create curved path using sine wave with random parameters
        x = t * self.length - self.length / 2
        y = curvature * np.sin(2 * np.pi * t + twist) * 0.1
        z = np.zeros_like(t) + 0.01  # Slightly above surface
        
        # Stack and add position offset
        self.points = np.stack([x, y, z], axis=1) + self.position
        
        return self.points
    
    def get_centerpoint(self) -> np.ndarray:
        """Get the centerpoint of the cable."""
        return np.mean(self.points, axis=0)
    
    def get_orientation(self) -> np.ndarray:
        """
        Calculate the primary orientation of the cable using PCA.
        Returns the principal direction vector.
        """
        centered_points = self.points - self.get_centerpoint()
        # Use SVD for PCA
        _, _, vt = np.linalg.svd(centered_points)
        principal_direction = vt[0]  # First principal component
        return principal_direction
    
    def get_tangent_at_point(self, point_idx: int) -> np.ndarray:
        """
        Get the tangent vector at a specific point on the cable.
        
        Args:
            point_idx: Index of the point
            
        Returns:
            Normalized tangent vector
        """
        if point_idx == 0:
            tangent = self.points[1] - self.points[0]
        elif point_idx == len(self.points) - 1:
            tangent = self.points[-1] - self.points[-2]
        else:
            tangent = self.points[point_idx + 1] - self.points[point_idx - 1]
        
        # Normalize
        return tangent / (np.linalg.norm(tangent) + 1e-8)
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the axis-aligned bounding box of the cable."""
        min_bounds = np.min(self.points, axis=0)
        max_bounds = np.max(self.points, axis=0)
        return min_bounds, max_bounds
