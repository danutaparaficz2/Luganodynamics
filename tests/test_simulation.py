"""Test suite for robotic cable pickup simulation."""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simulation.cable import Cable
from src.simulation.conveyor import ConveyorBelt
from src.simulation.camera import RGBDCamera
from src.robot.gripper import RoboticHand, GripperState
from src.simulation.environment import SimulationEnvironment


class TestCable(unittest.TestCase):
    """Test cable object functionality."""
    
    def test_cable_creation(self):
        """Test cable can be created with default parameters."""
        cable = Cable()
        self.assertEqual(cable.connector_length, 0.05)
        self.assertEqual(cable.cable_length, 0.3)
        
    def test_cable_position_update(self):
        """Test cable position can be updated."""
        cable = Cable()
        new_pos = np.array([1.0, 2.0, 3.0])
        new_ori = np.array([0.1, 0.2, 0.3])
        cable.update_position(new_pos, new_ori)
        np.testing.assert_array_equal(cable.position, new_pos)
        np.testing.assert_array_equal(cable.orientation, new_ori)
        
    def test_grasp_pose(self):
        """Test grasp pose calculation."""
        cable = Cable()
        cable.update_position(np.array([1.0, 0.0, 0.8]), np.array([0.0, 0.0, 0.0]))
        grasp_pos, grasp_ori = cable.get_grasp_pose()
        self.assertIsInstance(grasp_pos, np.ndarray)
        self.assertIsInstance(grasp_ori, np.ndarray)


class TestConveyorBelt(unittest.TestCase):
    """Test conveyor belt functionality."""
    
    def test_belt_creation(self):
        """Test conveyor belt can be created."""
        belt = ConveyorBelt()
        self.assertEqual(belt.length, 2.0)
        self.assertEqual(belt.speed, 0.05)
        
    def test_add_cable(self):
        """Test adding cable to belt."""
        belt = ConveyorBelt()
        cable = Cable()
        belt.add_cable(cable)
        self.assertEqual(len(belt.cables), 1)
        
    def test_belt_movement(self):
        """Test belt moves cables."""
        belt = ConveyorBelt()
        cable = Cable()
        belt.add_cable(cable, x_position=0.0)
        initial_x = cable.position[0]
        belt.update(1.0)  # 1 second
        self.assertGreater(cable.position[0], initial_x)


class TestRGBDCamera(unittest.TestCase):
    """Test RGB-D camera functionality."""
    
    def test_camera_creation(self):
        """Test camera can be created."""
        camera = RGBDCamera()
        self.assertEqual(camera.width, 640)
        self.assertEqual(camera.height, 480)
        
    def test_camera_intrinsics(self):
        """Test camera intrinsic matrix."""
        camera = RGBDCamera()
        K = camera.get_camera_matrix()
        self.assertEqual(K.shape, (3, 3))


class TestRoboticHand(unittest.TestCase):
    """Test robotic hand functionality."""
    
    def test_robot_creation(self):
        """Test robot can be created."""
        robot = RoboticHand()
        self.assertEqual(robot.gripper_state, GripperState.OPEN)
        
    def test_gripper_control(self):
        """Test gripper open/close."""
        robot = RoboticHand()
        robot.close_gripper()
        self.assertEqual(robot.gripper_state, GripperState.CLOSED)
        robot.open_gripper()
        self.assertEqual(robot.gripper_state, GripperState.OPEN)
        
    def test_robot_movement(self):
        """Test robot movement."""
        robot = RoboticHand()
        target = np.array([1.0, 0.0, 1.0])
        success = robot.move_to(target, np.array([0.0, 0.0, 0.0]))
        self.assertTrue(success)


class TestSimulationEnvironment(unittest.TestCase):
    """Test simulation environment."""
    
    def test_environment_creation(self):
        """Test environment can be created."""
        env = SimulationEnvironment()
        self.assertIsNotNone(env.conveyor)
        self.assertIsNotNone(env.camera)
        self.assertIsNotNone(env.robot)
        
    def test_add_cable(self):
        """Test adding cable to environment."""
        env = SimulationEnvironment()
        env.add_cable_to_belt()
        cables = env.conveyor.get_all_cables()
        self.assertEqual(len(cables), 1)
        
    def test_simulation_step(self):
        """Test simulation step."""
        env = SimulationEnvironment()
        state = env.step()
        self.assertIn('time', state)
        self.assertIn('robot_position', state)
        
    def test_reset(self):
        """Test environment reset."""
        env = SimulationEnvironment()
        env.add_cable_to_belt()
        env.step()
        env.reset()
        self.assertEqual(env.time, 0.0)


if __name__ == '__main__':
    unittest.main()
