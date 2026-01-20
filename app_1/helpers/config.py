"""
Configuration Module
====================
Tüm pipeline ayarlarını içerir.
"""

import torch


class Config:
    """Pipeline configuration."""
    
    # Model paths
    POSE_MODEL_PATH = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\yolo26l-pose.pt"
    COURT_MODEL_PATH = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\models\yolo\best_seg.pt"
    
    # OCR model
    PARSEQ_MODEL = "parseq_tiny"  # PARSeq: Scene Text Recognition
    
    # Input/Output paths
    INPUT_VIDEO = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\input\basketball_game.mp4"
    OUTPUT_DIR = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\output"
    CROPS_DIR = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\videos\output\jersey_crops"
    TEST_IMAGE_DIR = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\test"
    
    # Video input settings
    INPUT_WIDTH = 1920
    INPUT_HEIGHT = 1080
    SHOW_COURT_MASK = False  # Saha maskesini görselleştir
    SHOW_POSE_SKELETON = True  # Pose çizgilerini göster
    SHOW_SAM_MASK = True  # SAM maskelerini görselleştir
    USE_COURT_DETECTION = False  # Court kontrolünü kapat
    USE_CAMERA_COMPENSATION = True  # Kamera hareketi kompanzasyonu
    USE_SAM_SEGMENTATION = True  # SAM2 mask segmentasyonu
    SAM_MODEL = "sam2_b.pt"  # SAM model (sam2_s.pt, sam2_b.pt, sam2_l.pt)
    
    # Jersey crop settings
    JERSEY_EXPAND_RATIO = 0.5  # Daha geniş crop
    MIN_CROP_SIZE = 24  # Daha küçük crop'lar da kabul
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.25  # Düşürdük (0.4 -> 0.25)
    MIN_VISIBLE_KEYPOINTS = 4    # Düşürdük (8 -> 6)
    KEYPOINT_CONFIDENCE = 0.2    # Düşürdük (0.3 -> 0.2)
    MIN_OCR_CONFIDENCE = 0.60    # Düşürdük (0.70 -> 0.60)
    
    # Referee detection (HSV thresholds)
    REFEREE_SATURATION_THRESHOLD = 50
    REFEREE_VALUE_THRESHOLD = 120
    
    # Processing
    FRAME_SKIP = 3  # Her N frame'de 1 işle
    TRACKING_INTERVAL = 3  # Tracking her N frame'de 1 (interpolasyon)
    
    # Performance
    USE_HALF = False  # FP16 (dtype hatası nedeniyle kapalı)
    MAX_WORKERS = 2  # OCR thread sayısı
    YOLO_IMGSZ = 1088  # YOLO input boyutu (32'nin katı olmalı)
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
