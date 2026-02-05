"""
Homography Transformer
======================
Calculates and applies homography transformation from camera view to tactical bird's eye view.
"""

import cv2
import numpy as np
import sys
import os
from typing import Dict, Tuple

# Add reference repo to path to use its Utils if needed, 
# but better to re-implement core logic to be standalone.

def fix_homography_flip(H: np.ndarray) -> np.ndarray:
    """
    Detect and correct horizontal flip in the homography matrix.
    """
    R = H[:2, :2]
    if np.linalg.det(R) < 0:
        H[:, 0] *= -1
    return H

def blend_homographies(H1: np.ndarray, H2: np.ndarray, alpha: float) -> np.ndarray:
    """Average two homographies."""
    H_blend = alpha * H1 + (1 - alpha) * H2
    return H_blend / H_blend[-1, -1]

class HomographyTransformer:
    """
    Manages the perspective transformation from camera view to tactical view.
    """
    
    # Standard NBA Court Dimensions (in cm or relative units)
    # Mapping our 18 keypoints to top-down coordinates.
    # Ref: keys 0-17 from our custom trained model.
    # We need to manually map the 18 model/dataset keypoints to real world coords.
    # Based on the visual inspection of dataset: 
    # 0-3: Corners, 4-5: Center line edges, etc.
    
    # NBA Court: 94 feet x 50 feet (28.65m x 15.24m)
    # Tactical Map Size: 940 x 500 pixels (10px per foot)
    
    COURT_WIDTH = 940
    COURT_HEIGHT = 500
    
    # Define standard court keypoints (Top-down view coordinates)
    # Assuming the 18 keypoints from 'reloc2' dataset correspond to standard locations.
    # We need to verify the index mapping.
    # Typical mapping based on standard datasets:
    # 0: Top-Left Corner
    # 1: Top-Right Corner
    # 2: Bottom-Right Corner
    # 3: Bottom-Left Corner
    # 4: Center Line Top
    # 5: Center Line Bottom
    # ... (need to confirm mapping from dataset/visualization)
    
    # For now, we will use a generic mapping dictionary that should be tuned
    # based on the `test_keypoint.pt` output indices key.
    
    def __init__(self, alpha_smooth: float = 0.95):
        """More aggressive smoothing (0.95) for stability with dynamic cameras."""
        self.alpha = alpha_smooth
        self.prev_H = None
        self.prev_keypoints = None
        self.court_ref_points = self._get_court_reference_points()
        self.frame_shape = None
        
    def _get_court_reference_points(self) -> Dict[int, Tuple[int, int]]:
        """
        Define 2D coordinates for the 18 keypoints on the tactical map.
        Based on User's Evaluated Mapping:
        
        Left Baseline (x=0): 0(Top)..5(Bot)
        Left FT (x=190): 8(Top), 9(Bot)
        Center (x=470): 7(Top), 6(Bot)
        Right FT (x=750): 16(Top), 17(Bot)
        Right Baseline (x=940): 15(Top)..10(Bot)
        """
        w, h = self.COURT_WIDTH, self.COURT_HEIGHT
        
        points = {
            # === Left Side (Baseline x=0) ===
            0: (0, 0),          # Top-Left Corner
            1: (0, 40),         # Top 3pt
            2: (0, 170),        # Top Paint
            3: (0, 330),        # Bottom Paint
            4: (0, 460),        # Bottom 3pt
            5: (0, 500),        # Bottom-Left Corner
            
            # === Left Free Throw (x=190) ===
            8: (190, 170),      # Top FT
            9: (190, 330),      # Bottom FT
            
            # === Center Line (x=470) ===
            7: (470, 0),        # Top Center
            6: (470, 500),      # Bottom Center
            
            # === Right Free Throw (x=750) ===
            16: (750, 170),     # Top FT
            17: (750, 330),     # Bottom FT
            
            # === Right Baseline (x=940) ===
            15: (940, 0),       # Top-Right Corner
            14: (940, 40),      # Top 3pt
            13: (940, 170),     # Top Paint
            12: (940, 330),     # Bottom Paint
            11: (940, 460),     # Bottom 3pt
            10: (940, 500)      # Bottom-Right Corner
        }
        return points

    def update(self, detected_keypoints: np.ndarray) -> np.ndarray:
        """
        Update and return current homography matrix based on detected keypoints.
        """
        if detected_keypoints is None:
            return self.prev_H
            
        src_pts = []
        dst_pts = []
        
        # Higher confidence threshold for stability (0.5 instead of 0.3)
        for i, (x, y, conf) in enumerate(detected_keypoints):
            if conf > 0.5 and i in self.court_ref_points:
                src_pts.append([x, y])
                dst_pts.append(self.court_ref_points[i])
                
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)
        
        # Need at least 6 points for stable homography (was 4)
        if len(src_pts) < 6:
            return self.prev_H
            
        try:
            # Stricter RANSAC: threshold 3.0 (was 5.0), more iterations
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0, maxIters=2000)
            if H is None: 
                return self.prev_H
            
            # Check inlier count - need at least 5 inliers
            inlier_count = np.sum(mask) if mask is not None else 0
            if inlier_count < 5:
                return self.prev_H
            
            # Reprojection error check
            reproj = cv2.perspectiveTransform(src_pts.reshape(-1,1,2), H).reshape(-1, 2)
            errors = np.linalg.norm(reproj - dst_pts, axis=1)
            mean_error = np.mean(errors)
            if mean_error > 25:  # Too high error, reject this H
                return self.prev_H
            
            H = fix_homography_flip(H)
            
            # Check if H is reasonable (not too skewed)
            det = np.linalg.det(H[:2,:2])
            if abs(det) < 1e-5:
                return self.prev_H

            if self.prev_H is not None:
                H = blend_homographies(self.prev_H, H, self.alpha)
            self.prev_H = H
            return H
        except Exception:
            return self.prev_H


    def transform_points(self, points: np.ndarray, H: np.ndarray = None) -> np.ndarray:
        """
        Transform points (e.g., player feet) to tactical view.
        
        Args:
            points: (N, 2) array of [x, y]
            H: Optional override homography matrix
            
        Returns:
            transformed_points: (N, 2)
        """
        if H is None:
            H = self.prev_H
            
        if H is None or len(points) == 0:
            return points
            
        # Reshape for perspectiveTransform: (N, 1, 2)
        pts_in = points.reshape(-1, 1, 2).astype(np.float32)
        pts_out = cv2.perspectiveTransform(pts_in, H)
        return pts_out.reshape(-1, 2)

    def draw_court_lines_hybrid(self, frame: np.ndarray, keypoints: np.ndarray, H: np.ndarray = None, conf_threshold: float = 0.3) -> np.ndarray:
        """
        Hybrid approach for drawing court lines:
        - Use ACTUAL keypoint position if detected (confidence > threshold)
        - Use HOMOGRAPHY projection if keypoint is not visible
        
        This combines the accuracy of direct keypoints with the completeness of homography prediction.
        """
        if keypoints is None:
            return frame
            
        if H is None:
            H = self.prev_H
            
        frame_vis = frame.copy()
        
        # Get H_inv for predicting invisible points
        H_inv = None
        if H is not None:
            try:
                H_inv = np.linalg.inv(H)
            except:
                H_inv = None
        
        # Define which keypoints to connect
        lines = [
            # === Left Side ===
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (2, 8), (8, 9), (9, 3),
            
            # Center Line
            (7, 6),
            
            # === Right Side ===
            (15, 14), (14, 13), (13, 12), (12, 11), (11, 10),
            (13, 16), (16, 17), (17, 12),
            
            # === Sidelines ===
            (0, 7), (7, 15),
            (5, 6), (6, 10)
        ]
        
        def get_point(idx):
            """
            Get point position:
            - If keypoint is detected (conf > threshold), use it
            - Otherwise, use homography to predict from template
            """
            if idx < len(keypoints):
                x, y, c = keypoints[idx]
                if c > conf_threshold:
                    return (int(x), int(y)), True  # (point, is_real)
            
            # Keypoint not visible, try to predict using homography
            if H_inv is not None and idx in self.court_ref_points:
                tactical_pt = np.array([self.court_ref_points[idx]], dtype=np.float32).reshape(-1, 1, 2)
                camera_pt = cv2.perspectiveTransform(tactical_pt, H_inv).reshape(-1, 2)[0]
                if -1000 < camera_pt[0] < 3000 and -1000 < camera_pt[1] < 3000:
                    return (int(camera_pt[0]), int(camera_pt[1])), False
            
            return None, False
        
        for p1_idx, p2_idx in lines:
            pt1, is_real1 = get_point(p1_idx)
            pt2, is_real2 = get_point(p2_idx)
            
            if pt1 is not None and pt2 is not None:
                # Color: Blue if both real, Cyan if at least one is predicted
                if is_real1 and is_real2:
                    color = (255, 0, 0)  # Blue - both points are real
                else:
                    color = (255, 255, 0)  # Cyan - at least one predicted
                cv2.line(frame_vis, pt1, pt2, color, 3)
        
        return frame_vis
