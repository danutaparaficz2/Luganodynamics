"""Main simulation environment for robotic cable pickup."""

import numpy as np
from typing import Optional, List, Dict

# Support both relative and absolute imports
try:
    from .cable import Cable
    from .conveyor import ConveyorBelt
    from .camera import RGBDCamera
    from ..vision.cable_detector import CableDetector
    from ..robot.gripper import RoboticHand, GripperState
except ImportError:
    from simulation.cable import Cable
    from simulation.conveyor import ConveyorBelt
    from simulation.camera import RGBDCamera
    from vision.cable_detector import CableDetector
    from robot.gripper import RoboticHand, GripperState


class SimulationEnvironment:
    """Main simulation environment integrating all components.
    
    This class manages the complete simulation loop including:
    - Conveyor belt with cables
    - RGB-D camera
    - Cable detection and pose estimation
    - Robotic hand control
    - Training data collection
    """
    
    def __init__(
        self,
        time_step: float = 0.01,  # 10ms simulation steps
        use_gui: bool = False
    ):
        """Initialize simulation environment.
        
        Args:
            time_step: Simulation time step in seconds
            use_gui: Whether to enable visualization (if available)
        """
        self.time_step = time_step
        self.use_gui = use_gui
        
        # Create simulation components
        self.conveyor = ConveyorBelt()
        self.camera = RGBDCamera()
        self.robot = RoboticHand()
        self.detector = CableDetector(self.camera.get_camera_matrix())
        
        # Simulation state
        self.time = 0.0
        self.episode = 0
        self.total_pickups = 0
        self.successful_pickups = 0
        self.failed_pickups = 0
        
        # Current task state
        self.current_cable: Optional[Cable] = None
        self.pickup_in_progress = False
        self.pickup_phase = "idle"  # idle, approach, grasp, lift, return
        
    def reset(self):
        """Reset the simulation to initial state."""
        self.time = 0.0
        self.conveyor = ConveyorBelt()
        self.robot.return_home()
        self.current_cable = None
        self.pickup_in_progress = False
        self.pickup_phase = "idle"
        
    def add_cable_to_belt(
        self,
        cable: Optional[Cable] = None,
        x_position: Optional[float] = None
    ):
        """Add a cable to the conveyor belt.
        
        Args:
            cable: Cable to add (creates default if None)
            x_position: Initial position on belt (start if None)
        """
        if cable is None:
            cable = Cable()
        
        self.conveyor.add_cable(cable, x_position)
    
    def step(self) -> Dict:
        """Execute one simulation step.
        
        Returns:
            Dict with simulation state information
        """
        # Update conveyor belt
        self.conveyor.update(self.time_step)
        
        # Update robot
        robot_reached = self.robot.update(self.time_step)
        
        # Execute pickup behavior if in progress
        if self.pickup_in_progress:
            self._execute_pickup_phase(robot_reached)
        else:
            # Check if there's a cable at pickup position
            cable_at_pickup = self.conveyor.get_cable_at_pickup()
            if cable_at_pickup is not None:
                # Start pickup
                self._initiate_pickup(cable_at_pickup)
        
        # Update simulation time
        self.time += self.time_step
        
        # Return state info
        return self._get_state_info()
    
    def _initiate_pickup(self, cable: Cable):
        """Initiate pickup sequence for a cable.
        
        Args:
            cable: Cable to pick up
        """
        self.current_cable = cable
        self.pickup_in_progress = True
        self.pickup_phase = "detect"
        
    def _execute_pickup_phase(self, robot_reached: bool):
        """Execute current phase of pickup sequence.
        
        Args:
            robot_reached: Whether robot has reached its target
        """
        if self.pickup_phase == "detect":
            # Capture image and detect cable
            # Pass all cables in scene for rendering
            scene_cables = self.conveyor.get_all_cables()
            rgb, depth = self.camera.capture_rgbd(scene_cables)
            detections = self.detector.detect_cables(rgb, depth)
            
            if len(detections) > 0:
                # Estimate grasp pose
                grasp_pos, grasp_ori = self.detector.compute_grasp_pose(
                    detections[0], depth
                )
                
                # Move to approach position
                approach_pos = grasp_pos.copy()
                approach_pos[2] += 0.1  # 10cm above
                self.robot.move_to(approach_pos, grasp_ori)
                self.pickup_phase = "approach"
            else:
                # Failed to detect
                self._complete_pickup(success=False)
        
        elif self.pickup_phase == "approach":
            if robot_reached:
                # Move down to grasp position
                grasp_pos, grasp_ori = self.current_cable.get_grasp_pose()
                self.robot.move_to(grasp_pos, grasp_ori)
                self.pickup_phase = "grasp"
        
        elif self.pickup_phase == "grasp":
            if robot_reached:
                # Close gripper
                self.robot.close_gripper()
                self.pickup_phase = "lift"
        
        elif self.pickup_phase == "lift":
            if robot_reached:
                # Lift cable
                lift_pos = self.robot.position.copy()
                lift_pos[2] += 0.2  # Lift 20cm
                self.robot.move_to(lift_pos, self.robot.orientation)
                self.pickup_phase = "return"
        
        elif self.pickup_phase == "return":
            if robot_reached:
                # Return home
                self.robot.return_home()
                self.robot.open_gripper()
                # Remove cable from belt
                self.conveyor.remove_cable(self.current_cable)
                self._complete_pickup(success=True)
    
    def _complete_pickup(self, success: bool):
        """Complete pickup sequence.
        
        Args:
            success: Whether pickup was successful
        """
        self.pickup_in_progress = False
        self.pickup_phase = "idle"
        self.current_cable = None
        self.total_pickups += 1
        
        if success:
            self.successful_pickups += 1
        else:
            self.failed_pickups += 1
    
    def _get_state_info(self) -> Dict:
        """Get current simulation state information.
        
        Returns:
            Dict with state information
        """
        return {
            'time': self.time,
            'episode': self.episode,
            'robot_position': self.robot.position.copy(),
            'robot_orientation': self.robot.orientation.copy(),
            'gripper_state': self.robot.get_gripper_state().value,
            'cables_on_belt': len(self.conveyor.get_all_cables()),
            'pickup_in_progress': self.pickup_in_progress,
            'pickup_phase': self.pickup_phase,
            'total_pickups': self.total_pickups,
            'successful_pickups': self.successful_pickups,
            'failed_pickups': self.failed_pickups,
            'success_rate': (
                self.successful_pickups / self.total_pickups
                if self.total_pickups > 0 else 0.0
            )
        }
    
    def capture_observation(self) -> Dict:
        """Capture current observation (RGB-D image and robot state).
        
        Returns:
            Dict with observation data
        """
        scene_cables = self.conveyor.get_all_cables()
        rgb, depth = self.camera.capture_rgbd(scene_cables)
        
        return {
            'rgb_image': rgb,
            'depth_image': depth,
            'robot_position': self.robot.position.copy(),
            'robot_orientation': self.robot.orientation.copy(),
            'gripper_state': self.robot.get_gripper_state().value,
            'timestamp': self.time
        }
    
    def run_episode(
        self,
        num_cables: int = 5,
        max_steps: int = 10000
    ) -> Dict:
        """Run a complete episode with multiple cable pickups.
        
        Args:
            num_cables: Number of cables to add to belt
            max_steps: Maximum simulation steps
            
        Returns:
            Dict with episode statistics
        """
        self.reset()
        self.episode += 1
        
        # Add cables at intervals
        for i in range(num_cables):
            cable = Cable()
            self.add_cable_to_belt(cable, x_position=i * 0.4)
        
        # Run simulation
        for step in range(max_steps):
            state = self.step()
            
            # Check if all cables processed
            if (state['cables_on_belt'] == 0 and 
                not state['pickup_in_progress']):
                break
        
        return self._get_state_info()
    
    def get_training_data(self) -> List[Dict]:
        """Get collected training data.
        
        Returns:
            List of training samples
        """
        # TODO: Implement data collection during episodes
        return []
