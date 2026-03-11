"""
Configuration Module
====================
Tüm pipeline ayarlarını içerir.
"""

import os
import torch

# Project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Config:
    """Pipeline configuration."""
    
    # Model paths
    POSE_MODEL_PATH = os.path.join(ROOT_DIR, "yolo26m-pose.pt")
    
    # OCR model
    PARSEQ_MODEL = "parseq"  # PARSeq: Scene Text Recognition
    
    # Input/Output paths
    INPUT_VIDEO = os.path.join(ROOT_DIR, "videos", "input", "basketball_game.mp4")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "videos", "output")
    CROPS_DIR = os.path.join(ROOT_DIR, "videos", "output", "jersey_crops")
    TEST_IMAGE_DIR = os.path.join(ROOT_DIR, "test")
    
    # Jersey crop settings
    JERSEY_EXPAND_RATIO = 0.4
    MIN_CROP_SIZE = 32
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.4
    MIN_VISIBLE_KEYPOINTS = 8
    KEYPOINT_CONFIDENCE = 0.3
    MIN_OCR_CONFIDENCE = 0.70
    
    # Referee detection (HSV thresholds)
    REFEREE_SATURATION_THRESHOLD = 50
    REFEREE_VALUE_THRESHOLD = 120
    
    # Processing
    FRAME_SKIP = 3
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

