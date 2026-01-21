"""Cable detection and pose estimation from RGB-D images."""

import numpy as np
import cv2
from typing import Tuple, Optional, List


class CableDetector:
    """Detects cables and estimates their pose from RGB-D images.
    
    Uses computer vision techniques to identify elongated objects (cables)
    and estimate their 3D pose for grasping.
    """
    
    def __init__(
        self,
        camera_matrix: np.ndarray,
        depth_threshold: float = 0.05,  # 5cm above belt surface
        min_cable_length: float = 0.03,  # Minimum 3cm cable length to detect
        min_cable_width: float = 0.01    # Minimum 1cm cable width to detect
    ):
        """Initialize cable detector.
        
        Args:
            camera_matrix: Camera intrinsic matrix K (3x3)
            depth_threshold: Height threshold above belt for cable detection
            min_cable_length: Minimum cable length to detect in meters
            min_cable_width: Minimum cable width to detect in meters
        """
        self.camera_matrix = camera_matrix
        self.depth_threshold = depth_threshold
        self.min_cable_length = min_cable_length
        self.min_cable_width = min_cable_width
        
    def detect_cables(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray
    ) -> List[dict]:
        """Detect cables in RGB-D image.
        
        Args:
            rgb_image: RGB image (H, W, 3)
            depth_image: Depth map (H, W) in meters
            
        Returns:
            List of detected cables, each as a dict with:
            - 'bbox': Bounding box [x, y, w, h] in pixels
            - 'mask': Binary mask of cable region
            - 'center': Center point [u, v] in pixels
            - 'orientation': Estimated orientation angle in radians
        """
        # Convert to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        # Simple thresholding to find objects above belt
        # (In a real implementation, this would use the depth image)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        detected_cables = []
        
        for contour in contours:
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < 100:  # Minimum area threshold
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if elongated (cable-like)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            if aspect_ratio < 1.5:  # Must be somewhat elongated
                continue
            
            # Create mask for this cable
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # Calculate center
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            # Estimate orientation using PCA or minimum area rectangle
            rect = cv2.minAreaRect(contour)
            orientation = np.radians(rect[2])  # Angle in radians
            
            detected_cables.append({
                'bbox': [x, y, w, h],
                'mask': mask,
                'center': [cx, cy],
                'orientation': orientation,
                'contour': contour
            })
        
        return detected_cables
    
    def estimate_3d_pose(
        self,
        cable_detection: dict,
        depth_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate 3D pose of cable from detection and depth.
        
        Args:
            cable_detection: Cable detection dict from detect_cables()
            depth_image: Depth map (H, W) in meters
            
        Returns:
            Tuple of (position, orientation) where:
            - position: 3D position [x, y, z] in meters
            - orientation: 3D orientation [roll, pitch, yaw] in radians
        """
        # Get cable center in image
        cx, cy = cable_detection['center']
        
        # Get depth at cable center
        depth = depth_image[cy, cx]
        
        # Back-project to 3D using camera intrinsics
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx_cam = self.camera_matrix[0, 2]
        cy_cam = self.camera_matrix[1, 2]
        
        # 3D position in camera coordinates
        x_cam = (cx - cx_cam) * depth / fx
        y_cam = (cy - cy_cam) * depth / fy
        z_cam = depth
        
        position = np.array([x_cam, y_cam, z_cam])
        
        # Estimate orientation
        # For now, use 2D orientation with zero roll and pitch
        yaw = cable_detection['orientation']
        orientation = np.array([0.0, 0.0, yaw])
        
        return position, orientation
    
    def compute_grasp_pose(
        self,
        cable_detection: dict,
        depth_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute optimal grasp pose for a detected cable.
        
        Args:
            cable_detection: Cable detection dict
            depth_image: Depth map in meters
            
        Returns:
            Tuple of (grasp_position, grasp_orientation) in 3D space
        """
        # Estimate 3D pose first
        position, orientation = self.estimate_3d_pose(cable_detection, depth_image)
        
        # Grasp position is at estimated cable position
        # Add small offset for gripper approach
        grasp_position = position.copy()
        grasp_position[2] += 0.05  # 5cm above for approach
        
        # Grasp orientation aligned with cable
        grasp_orientation = orientation.copy()
        
        return grasp_position, grasp_orientation
