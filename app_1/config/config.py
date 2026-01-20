"""
Configuration Module
====================
Tüm pipeline ayarlarını içerir.
"""

import torch


class Config:
    """Pipeline configuration."""
    
    # Model paths
    POSE_MODEL_PATH = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\yolo26m-pose.pt"
    
    # OCR model
    PARSEQ_MODEL = "parseq"  # PARSeq: Scene Text Recognition
    
    # Input/Output paths
    INPUT_VIDEO = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\input\basketball_game.mp4"
    OUTPUT_DIR = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\output"
    CROPS_DIR = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\output\jersey_crops"
    TEST_IMAGE_DIR = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\test"
    
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
