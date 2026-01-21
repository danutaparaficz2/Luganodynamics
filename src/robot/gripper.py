"""Robotic hand controller for cable pickup."""

import numpy as np
from typing import Tuple, Optional
from enum import Enum


class GripperState(Enum):
    """States of the gripper."""
    OPEN = "open"
    CLOSED = "closed"
    MOVING = "moving"


class RoboticHand:
    """Simulates a robotic hand/gripper for picking up cables.
    
    The hand has position, orientation, and gripper state.
    """
    
    def __init__(
        self,
        home_position: np.ndarray = np.array([1.0, 0.0, 1.5]),  # Above belt
        gripper_width: float = 0.08,  # 8cm gripper opening
        max_reach: float = 0.8,       # Maximum reach distance
        speed: float = 0.2            # Movement speed m/s
    ):
        """Initialize robotic hand.
        
        Args:
            home_position: Home position [x, y, z] in meters
            gripper_width: Maximum gripper opening width in meters
            max_reach: Maximum reach distance from base in meters
            speed: Movement speed in m/s
        """
        self.home_position = np.array(home_position)
        self.gripper_width = gripper_width
        self.max_reach = max_reach
        self.speed = speed
        
        # Current state
        self.position = self.home_position.copy()
        self.orientation = np.array([0.0, 0.0, 0.0])  # [roll, pitch, yaw]
        self.gripper_state = GripperState.OPEN
        self.gripper_opening = gripper_width  # Current opening
        
        # Target state (for motion planning)
        self.target_position = None
        self.target_orientation = None
        
    def move_to(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray
    ) -> bool:
        """Set target position and orientation for the hand.
        
        Args:
            target_position: Target position [x, y, z]
            target_orientation: Target orientation [roll, pitch, yaw]
            
        Returns:
            True if target is reachable, False otherwise
        """
        # Check if target is within reach
        distance = np.linalg.norm(target_position[:2] - self.home_position[:2])
        if distance > self.max_reach:
            return False
        
        self.target_position = np.array(target_position)
        self.target_orientation = np.array(target_orientation)
        self.gripper_state = GripperState.MOVING
        
        return True
    
    def update(self, dt: float) -> bool:
        """Update hand state (move towards target).
        
        Args:
            dt: Time step in seconds
            
        Returns:
            True if target reached, False if still moving
        """
        if self.target_position is None:
            return True
        
        # Calculate direction to target
        direction = self.target_position - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 0.001:  # Close enough
            self.position = self.target_position.copy()
            self.orientation = self.target_orientation.copy()
            self.target_position = None
            self.target_orientation = None
            if self.gripper_state == GripperState.MOVING:
                self.gripper_state = GripperState.OPEN
            return True
        
        # Move towards target
        move_distance = min(self.speed * dt, distance)
        self.position += (direction / distance) * move_distance
        
        # Interpolate orientation (simple linear for now)
        ori_diff = self.target_orientation - self.orientation
        self.orientation += ori_diff * min(1.0, self.speed * dt / distance)
        
        return False
    
    def open_gripper(self):
        """Open the gripper."""
        self.gripper_state = GripperState.OPEN
        self.gripper_opening = self.gripper_width
    
    def close_gripper(self):
        """Close the gripper to grasp."""
        self.gripper_state = GripperState.CLOSED
        self.gripper_opening = 0.01  # Small opening when closed
    
    def grasp_object(
        self,
        object_position: np.ndarray,
        object_orientation: np.ndarray
    ) -> bool:
        """Execute grasp sequence for an object.
        
        Args:
            object_position: Position of object to grasp
            object_orientation: Orientation of object
            
        Returns:
            True if grasp initiated successfully
        """
        # Calculate approach position (above object)
        approach_position = object_position.copy()
        approach_position[2] += 0.1  # 10cm above
        
        # Set target to approach position
        success = self.move_to(approach_position, object_orientation)
        
        return success
    
    def return_home(self):
        """Return to home position."""
        self.move_to(self.home_position, np.array([0.0, 0.0, 0.0]))
    
    def get_gripper_state(self) -> GripperState:
        """Get current gripper state.
        
        Returns:
            Current gripper state
        """
        return self.gripper_state
    
    def is_grasping(self) -> bool:
        """Check if gripper is currently grasping.
        
        Returns:
            True if gripper is closed
        """
        return self.gripper_state == GripperState.CLOSED
    
    def compute_inverse_kinematics(
        self,
        target_position: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compute inverse kinematics for target position.
        
        This is a simplified version. In a real robot, this would compute
        joint angles to reach the target position.
        
        Args:
            target_position: Desired end-effector position
            
        Returns:
            Joint angles (simplified), or None if unreachable
        """
        # Simplified: just check reachability
        distance = np.linalg.norm(target_position[:2] - self.home_position[:2])
        if distance > self.max_reach:
            return None
        
        # Return dummy joint angles (in a real system, solve IK)
        # For now, just return position as "joint values"
        return target_position
