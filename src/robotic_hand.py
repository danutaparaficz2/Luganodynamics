"""
Robotic hand/gripper module for cable pickup.
"""
import numpy as np
from typing import Tuple, Optional
from src.cable import Cable


class GraspPose:
    """
    Represents a grasp pose for picking up a cable.
    """
    
    def __init__(self, position: np.ndarray, orientation: np.ndarray, 
                 approach_vector: np.ndarray, grip_width: float):
        """
        Initialize a grasp pose.
        
        Args:
            position: 3D position where to grasp
            orientation: Orientation vector of the cable at grasp point
            approach_vector: Direction from which to approach the grasp
            grip_width: Width of the gripper opening needed
        """
        self.position = position
        self.orientation = orientation
        self.approach_vector = approach_vector
        self.grip_width = grip_width
        self.confidence = 0.0  # Confidence score for this grasp
        
    def __repr__(self):
        return (f"GraspPose(pos={self.position}, orient={self.orientation}, "
                f"approach={self.approach_vector}, width={self.grip_width:.3f}, "
                f"confidence={self.confidence:.3f})")


class RoboticHand:
    """
    Represents a robotic hand/gripper for picking up cables.
    """
    
    def __init__(self, position: Tuple[float, float, float] = (0.0, 0.0, 0.5),
                 max_grip_width: float = 0.15, min_grip_width: float = 0.02):
        """
        Initialize a robotic hand.
        
        Args:
            position: Initial position (x, y, z) of the hand
            max_grip_width: Maximum gripper opening width
            min_grip_width: Minimum gripper opening width
        """
        self.position = np.array(position, dtype=np.float32)
        self.max_grip_width = max_grip_width
        self.min_grip_width = min_grip_width
        self.current_grip_width = max_grip_width
        self.is_gripping = False
        self.grasped_cable = None
        
    def compute_grasp_pose(self, cable: Cable) -> Optional[GraspPose]:
        """
        Compute an optimal grasp pose for a given cable.
        
        The strategy is to:
        1. Find the centerpoint of the cable
        2. Determine the cable's primary orientation
        3. Calculate an approach vector (perpendicular to cable, from above)
        4. Compute required grip width
        
        Args:
            cable: The cable to grasp
            
        Returns:
            GraspPose object or None if no valid grasp found
        """
        # Get cable centerpoint and orientation
        grasp_position = cable.get_centerpoint()
        cable_orientation = cable.get_orientation()
        
        # Ensure z-component is set to grasp height (slightly above cable)
        grasp_position[2] = cable.points[:, 2].mean() + 0.02
        
        # Compute approach vector (come from above, perpendicular to cable)
        # Approach direction should be downward but aligned with cable shape
        approach_vector = np.array([0.0, 0.0, -1.0])
        
        # Calculate required grip width based on cable thickness and orientation
        # For a cable, we estimate based on bounding box in perpendicular plane
        min_bounds, max_bounds = cable.get_bounding_box()
        cable_span = np.linalg.norm(max_bounds[:2] - min_bounds[:2])
        
        # Required grip width should accommodate cable span plus safety margin
        required_grip_width = min(cable_span * 0.5 + 0.02, self.max_grip_width)
        
        if required_grip_width < self.min_grip_width:
            required_grip_width = self.min_grip_width
        
        # Create grasp pose
        grasp_pose = GraspPose(
            position=grasp_position,
            orientation=cable_orientation,
            approach_vector=approach_vector,
            grip_width=required_grip_width
        )
        
        # Compute confidence score based on cable stability and orientation
        grasp_pose.confidence = self._compute_grasp_confidence(cable, grasp_pose)
        
        return grasp_pose
    
    def _compute_grasp_confidence(self, cable: Cable, grasp_pose: GraspPose) -> float:
        """
        Compute a confidence score for a grasp pose.
        
        Higher confidence for:
        - Cables with less curvature
        - Grasps at stable centerpoints
        - Appropriate grip widths
        
        Args:
            cable: The cable to grasp
            grasp_pose: The proposed grasp pose
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 1.0
        
        # Penalize if grip width is near limits
        if grasp_pose.grip_width > self.max_grip_width * 0.9:
            confidence *= 0.7
        
        # Reward grasping near the center of the cable
        cable_length = cable.length
        distances = np.linalg.norm(cable.points - grasp_pose.position, axis=1)
        min_distance = np.min(distances)
        if min_distance > 0.05:  # More than 5cm from cable
            confidence *= 0.5
        
        # Check cable curvature - straighter cables are easier to grasp
        point_distances = np.linalg.norm(np.diff(cable.points, axis=0), axis=1)
        curvature_variation = np.std(point_distances)
        if curvature_variation > 0.01:
            confidence *= 0.8
        
        return np.clip(confidence, 0.0, 1.0)
    
    def execute_grasp(self, grasp_pose: GraspPose, cable: Cable) -> bool:
        """
        Execute a grasp at the given pose.
        
        Args:
            grasp_pose: The grasp pose to execute
            cable: The cable to grasp
            
        Returns:
            True if grasp was successful, False otherwise
        """
        # Move to grasp position
        self.position = grasp_pose.position.copy()
        
        # Set grip width
        self.current_grip_width = grasp_pose.grip_width
        
        # Attempt grasp based on confidence
        success_threshold = 0.5
        if grasp_pose.confidence >= success_threshold:
            self.is_gripping = True
            self.grasped_cable = cable
            return True
        else:
            self.is_gripping = False
            return False
    
    def release(self):
        """Release the currently grasped cable."""
        self.is_gripping = False
        self.grasped_cable = None
        self.current_grip_width = self.max_grip_width
    
    def move_to(self, position: np.ndarray):
        """Move the hand to a new position."""
        self.position = position.copy()
        
    def get_state(self) -> dict:
        """Get the current state of the robotic hand."""
        return {
            'position': self.position.copy(),
            'grip_width': self.current_grip_width,
            'is_gripping': self.is_gripping,
            'has_cable': self.grasped_cable is not None
        }
