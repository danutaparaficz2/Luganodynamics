"""Camera simulation for RGB-D image capture."""

import numpy as np
from typing import Tuple, Optional
import cv2


class RGBDCamera:
    """Simulates an RGB-D camera mounted overhead.
    
    Generates synthetic RGB and depth images of the scene below.
    """
    
    def __init__(
        self,
        position: np.ndarray = np.array([1.0, 0.0, 2.0]),  # Position above belt
        look_at: np.ndarray = np.array([1.0, 0.0, 0.8]),   # Look at belt surface
        fov: float = 60.0,           # Field of view in degrees
        width: int = 640,            # Image width in pixels
        height: int = 480,           # Image height in pixels
        near: float = 0.1,           # Near clipping plane
        far: float = 5.0             # Far clipping plane
    ):
        """Initialize RGB-D camera.
        
        Args:
            position: Camera position [x, y, z] in meters
            look_at: Point camera is looking at [x, y, z]
            fov: Field of view in degrees
            width: Image width in pixels
            height: Image height in pixels
            near: Near clipping plane in meters
            far: Far clipping plane in meters
        """
        self.position = np.array(position)
        self.look_at = np.array(look_at)
        self.fov = fov
        self.width = width
        self.height = height
        self.near = near
        self.far = far
        
        # Calculate camera intrinsic parameters
        self._compute_intrinsics()
        
        # Calculate camera extrinsic parameters (pose)
        self._compute_extrinsics()
        
    def _compute_intrinsics(self):
        """Compute camera intrinsic matrix."""
        # Focal length in pixels
        focal_length_px = (self.width / 2.0) / np.tan(np.radians(self.fov / 2.0))
        
        # Principal point (center of image)
        cx = self.width / 2.0
        cy = self.height / 2.0
        
        # Intrinsic matrix
        self.K = np.array([
            [focal_length_px, 0, cx],
            [0, focal_length_px, cy],
            [0, 0, 1]
        ])
        
    def _compute_extrinsics(self):
        """Compute camera extrinsic matrix (rotation and translation)."""
        # Camera coordinate frame
        # Z-axis: from camera to look_at point
        z_axis = self.look_at - self.position
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # X-axis: perpendicular to Z and world up (assuming world up is [0, 0, 1])
        world_up = np.array([0, 0, 1])
        x_axis = np.cross(world_up, z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            # Camera looking straight up or down
            x_axis = np.array([1, 0, 0])
        else:
            x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y-axis: perpendicular to X and Z
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Rotation matrix (world to camera)
        self.R = np.vstack([x_axis, y_axis, z_axis])
        
        # Translation vector
        self.t = -self.R @ self.position
        
    def capture_rgbd(self, scene_objects) -> Tuple[np.ndarray, np.ndarray]:
        """Capture RGB-D image of the scene.
        
        Args:
            scene_objects: List of objects in the scene (cables, belt, etc.)
            
        Returns:
            Tuple of (rgb_image, depth_image)
            - rgb_image: RGB image of shape (height, width, 3), uint8
            - depth_image: Depth map of shape (height, width), float32, in meters
        """
        # Create synthetic images
        rgb_image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 200
        depth_image = np.ones((self.height, self.width), dtype=np.float32) * self.far
        
        # Draw belt
        belt_color = np.array([50, 50, 50], dtype=np.uint8)
        rgb_image[:, :] = belt_color
        
        # Simple depth for belt surface (distance from camera to belt)
        belt_depth = np.linalg.norm(self.position - np.array([1.0, 0.0, 0.8]))
        depth_image[:, :] = belt_depth
        
        # Render cables in the scene
        for obj in scene_objects:
            # Check if it's a cable (has connector_length attribute)
            if hasattr(obj, 'connector_length'):
                self._render_cable(obj, rgb_image, depth_image)
        
        return rgb_image, depth_image
    
    def _render_cable(self, cable, rgb_image: np.ndarray, depth_image: np.ndarray):
        """Render a cable in the RGB-D image.
        
        Args:
            cable: Cable object to render
            rgb_image: RGB image to draw into
            depth_image: Depth image to update
        """
        # Project cable connector to image
        connector_center = cable.position.copy()
        
        # Project multiple points of the cable
        points_2d = []
        for i in range(5):  # Sample a few points
            offset = np.array([i * cable.connector_length / 4, 0, 0])
            point_3d = connector_center + offset
            point_2d = self.project_point(point_3d)
            if point_2d is not None:
                points_2d.append(point_2d)
        
        if len(points_2d) == 0:
            return
        
        # Draw cable as an elongated region
        cable_color = np.array([200, 150, 100], dtype=np.uint8)  # Brownish color
        
        # Simple rectangular rendering
        if len(points_2d) >= 2:
            x_coords = [p[0] for p in points_2d]
            y_coords = [p[1] for p in points_2d]
            
            x_min = max(0, min(x_coords) - 10)
            x_max = min(self.width - 1, max(x_coords) + 10)
            y_min = max(0, min(y_coords) - 10)
            y_max = min(self.height - 1, max(y_coords) + 10)
            
            # Draw cable region
            rgb_image[y_min:y_max, x_min:x_max] = cable_color
            
            # Update depth (cable is above belt)
            cable_depth = np.linalg.norm(self.position - connector_center)
            depth_image[y_min:y_max, x_min:x_max] = cable_depth
    
    def project_point(self, point_3d: np.ndarray) -> Optional[Tuple[int, int]]:
        """Project a 3D point to 2D image coordinates.
        
        Args:
            point_3d: 3D point in world coordinates [x, y, z]
            
        Returns:
            Tuple of (u, v) pixel coordinates, or None if point is outside image
        """
        # Transform point to camera coordinates
        point_cam = self.R @ point_3d + self.t
        
        # Check if point is in front of camera
        if point_cam[2] <= 0:
            return None
            
        # Project to image plane
        point_2d_homogeneous = self.K @ point_cam
        u = int(point_2d_homogeneous[0] / point_2d_homogeneous[2])
        v = int(point_2d_homogeneous[1] / point_2d_homogeneous[2])
        
        # Check if point is within image bounds
        if 0 <= u < self.width and 0 <= v < self.height:
            return (u, v)
        else:
            return None
    
    def get_camera_matrix(self) -> np.ndarray:
        """Get camera intrinsic matrix.
        
        Returns:
            3x3 camera intrinsic matrix K
        """
        return self.K.copy()
    
    def get_camera_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get camera pose (rotation and translation).
        
        Returns:
            Tuple of (R, t) where:
            - R: 3x3 rotation matrix (world to camera)
            - t: 3x1 translation vector
        """
        return self.R.copy(), self.t.copy()
