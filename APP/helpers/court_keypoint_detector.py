"""
Court Keypoint Detector
=======================
Custom training YOLO-Pose model approach for basketball court keypoint detection.
Includes temporal smoothing and confidence filtering.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from collections import deque


class KeypointSmoother:
    """Temporal smoothing for keypoint stability using Exponential Moving Average (EMA)."""
    
    def __init__(self, num_keypoints: int = 18, alpha: float = 0.6):
        """
        Args:
            num_keypoints: Number of keypoints to track
            alpha: EMA weight for current frame (0-1). 
                   Lower = more smoothing (less jitter, more lag)
                   Higher = less smoothing (more responsive)
        """
        self.alpha = alpha
        self.smoothed = None
    
    def update(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Apply smoothing to keypoints.
        
        Args:
            keypoints: (N, 3) array [x, y, conf]
        """
        if keypoints is None:
            return self.smoothed
            
        if self.smoothed is None:
            self.smoothed = keypoints.copy()
        else:
            # Only update smoothed position if we have a confident detection
            # or if we don't have a confident previous position?
            # Simple EMA:
            
            # Vectorized update
            # We want to trust high confidence detections more?
            # For now standard EMA on coordinates, copy confidence
            
            # Mask for valid detections (some confidence)
            mask = keypoints[:, 2] > 0.0
            
            # Update valid keypoints
            self.smoothed[mask, :2] = (
                self.alpha * keypoints[mask, :2] + 
                (1 - self.alpha) * self.smoothed[mask, :2]
            )
            # Update confidence directly
            self.smoothed[mask, 2] = keypoints[mask, 2]
            
        return self.smoothed
    
    def reset(self):
        self.smoothed = None


class CourtKeypointDetector:
    """
    Basketball court keypoint detector using custom YOLO-Pose model.
    Detects 18 keypoints: corners, paint, center circle, etc.
    """
    
    def __init__(
        self,
        model_path: str = "models/keypoints/test_keypoint.pt",
        imgsz: int = 768,
        conf_threshold: float = 0.5,
        use_smoothing: bool = True
    ):
        """
        Args:
            model_path: Path to .pt model file
            imgsz: Inference resolution (should match training, e.g. 768 or 832)
            conf_threshold: Minimum confidence to consider a keypoint valid
            use_smoothing: Enable temporal smoothing
        """
        print(f"📦 Loading Court Keypoint Model: {model_path} (imgsz={imgsz})")
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        
        self.smoother = KeypointSmoother(num_keypoints=18, alpha=0.6) if use_smoothing else None
    
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect keypoints in a single frame.
        
        Returns:
            keypoints: (18, 3) array [x, y, conf] or None
        """
        results = self.model(frame, imgsz=self.imgsz, verbose=False)
        
        if not results or results[0].keypoints is None:
            return None
            
        # Get first detection (assuming single court)
        if len(results[0].keypoints.data) == 0:
            return None
            
        kpts = results[0].keypoints.data[0].cpu().numpy()  # (18, 3)
        
        # Smooth
        if self.smoother:
            kpts = self.smoother.update(kpts)
            
        return kpts
    
    def get_valid_keypoints(self, keypoints: np.ndarray) -> Dict[int, Tuple[int, int]]:
        """
        Filter keypoints by confidence threshold.
        
        Returns:
            Dictionary {index: (x, y)} for valid keypoints
        """
        if keypoints is None:
            return {}
            
        valid = {}
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > self.conf_threshold:
                valid[i] = (int(x), int(y))
        return valid

    def draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Draw keypoints on frame for visualization."""
        if keypoints is None:
            return frame
            
        out = frame.copy()
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > self.conf_threshold:
                # Green circle for keypoint
                cv2.circle(out, (int(x), int(y)), 8, (0, 255, 0), -1)
                
                # Label
                # Draw black border for contrast
                text = str(i)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1.0 # Larger scale
                thick = 3
                
                # Black Outline
                cv2.putText(out, text, (int(x)+10, int(y)-10), 
                           font, scale, (0, 0, 0), thick+2)
                # White Text
                cv2.putText(out, text, (int(x)+10, int(y)-10), 
                           font, scale, (255, 255, 255), thick)
        return out
